import os
import yaml
import numpy as np
import torch
import torch.nn as nn

from MachineLearning import DenseNN, ResDenseNN, LSTMNN, RNN
torch.set_default_dtype(torch.float64)

_ACTIVATION_MAP = {
    'ReLU':    nn.ReLU,
    'Tanh':    nn.Tanh,
    'Sigmoid': nn.Sigmoid,
}


def _is_cupy_array(obj):
    """True for CuPy ndarrays (not PyTorch tensors)."""
    if isinstance(obj, torch.Tensor):
        return False
    return type(obj).__module__.startswith('cupy')


class SurrogateModel:
    """
    Generic loader and predictor for any pretrained Lorenz surrogate model.

    Supports all four model types (DenseNN, ResDenseNN, LSTMNN, RNN)
    saved by Main_ML.py.  Handles normalisation internally so the public API
    operates entirely in physical space.

    Parameters
    ----------
    pth_path : str
        Path to a .pth checkpoint (full checkpoint from save_model(), or a bare
        state_dict from a _best_model.pth file).
    device : str or torch.device, optional
        Target device.  Defaults to CUDA if available, else CPU.
    yml_path : str, optional
        Explicit path to the .yml sidecar.  If None the class tries to
        auto-discover it next to the .pth file.

    Examples
    --------
    >>> surrogate = SurrogateModel('models/RNN_L63_trial1_1774989331.pth')
    >>> history = np.random.randn(4, 3)          # (prev_time_steps, N)
    >>> next_state = surrogate.predict(history)   # (N,)
    >>> traj = surrogate.rollout(history, 100)    # (100, N)
    """

    def __init__(self, pth_path, device=None, yml_path=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        checkpoint, yml_meta = self._load_files(pth_path, yml_path)

        is_full_checkpoint = 'model_state_dict' in checkpoint
        state_dict = checkpoint['model_state_dict'] if is_full_checkpoint else checkpoint

        # Merge metadata: checkpoint architecture + YAML sidecar
        arch = checkpoint.get('architecture', {}) if is_full_checkpoint else {}
        meta = {**yml_meta, **arch}  # checkpoint architecture takes priority

        if 'model_type' not in meta:
            raise ValueError(
                f"Cannot determine model_type from checkpoint or YAML sidecar. "
                f"Provide a yml_path or use a full checkpoint."
            )

        self.model_type = meta['model_type']
        self.input_size = meta['input_size']
        self.prev_time_steps = meta['prev_time_steps']
        self.metadata = meta

        # Resolve normalisation stats: checkpoint > YAML
        self.train_mean = self._resolve_array(
            checkpoint.get('train_mean') if is_full_checkpoint else None,
            yml_meta.get('train_mean'),
        )
        self.train_std = self._resolve_array(
            checkpoint.get('train_std') if is_full_checkpoint else None,
            yml_meta.get('train_std'),
        )

        # Build and load model
        self.model = self._build_model(meta, yml_meta)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._hidden = None  # recurrent hidden state (LSTMNN/RNN only)
        self._train_mean_t = None
        self._train_std_t = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, state_history):
        """
        Single-step prediction in physical space.

        Parameters
        ----------
        state_history : np.ndarray
            Shape ``(prev_time_steps, N)`` or ``(prev_time_steps * N,)``,
            ordered oldest-first: ``[s_{t-k+1}, ..., s_t]``.

        Returns
        -------
        np.ndarray of shape ``(N,)`` — the predicted next state.
        """
        state_history = np.asarray(state_history, dtype=np.float64)
        if state_history.ndim == 1:
            state_history = state_history.reshape(self.prev_time_steps, self.input_size)

        normalised = (state_history - self.train_mean) / self.train_std
        flat = normalised.reshape(1, -1)
        dtype = next(self.model.parameters()).dtype
        x = torch.as_tensor(flat, dtype=dtype, device=self.device)

        with torch.no_grad():
            if isinstance(self.model, (LSTMNN, RNN)):
                pred, self._hidden = self.model(x, self._hidden)
            else:
                pred = self.model(x)
            pred = pred.cpu().numpy().astype(np.float64).squeeze(0)

        return pred * self.train_std + self.train_mean

    def rollout(self, state_history, num_steps):
        """
        Multi-step autoregressive prediction in physical space.

        Each step's output is denormalised, appended to the window, and
        renormalised before the next forward pass.

        Parameters
        ----------
        state_history : np.ndarray
            Shape ``(prev_time_steps, N)``, ordered oldest-first.
        num_steps : int
            Number of future steps to predict.

        Returns
        -------
        np.ndarray of shape ``(num_steps, N)`` — predicted trajectory.
        """
        state_history = np.asarray(state_history, dtype=np.float64)
        if state_history.ndim == 1:
            state_history = state_history.reshape(self.prev_time_steps, self.input_size)

        self.reset_hidden()  # clear hidden state for fresh rollout
        window = state_history.copy()  # (prev_time_steps, N) in physical space
        predictions = []

        for _ in range(num_steps):
            pred = self.predict(window)       # (N,) physical space
            predictions.append(pred)
            window = np.concatenate([window[1:], pred[np.newaxis, :]], axis=0)

        return np.stack(predictions, axis=0)

    def __call__(self, state_history):
        """Alias for :meth:`predict`."""
        return self.predict(state_history)

    def reset_hidden(self, batch_size=None):
        """Reset hidden state for recurrent models. No-op for feedforward."""
        self._hidden = None

    def batch_predict(self, state_histories):
        """
        Batched single-step prediction in physical space.

        Runs B trajectories through the model in a single forward pass so the
        GPU can batch the matmuls. Stateless: never reads or writes
        ``self._hidden`` — each call is independent of any previous
        ``predict`` / ``rollout`` / ``batch_*`` call.

        For ``LSTMNN`` / ``RNN``, PyTorch's recurrent modules keep the hidden
        tensor shaped ``(num_layers, B, hidden_size)``; each of the B batch
        slots evolves independently, so trajectories never share memory.

        Parameters
        ----------
        state_histories : np.ndarray
            Shape ``(B, prev_time_steps, N)`` or ``(B, prev_time_steps * N)``,
            physical space, oldest-first along the time axis.

        Returns
        -------
        np.ndarray of shape ``(B, N)``.
        """
        window = self._prepare_batch_tensor(state_histories)

        with torch.inference_mode():
            pred_norm, _ = self._forward_normalized(window, hidden=None)

        return self._denorm_tensor(pred_norm).cpu().numpy().astype(np.float64)

    def batch_rollout(self, state_histories, num_steps):
        """
        Batched autoregressive rollout in physical space.

        All B members advance together; the entire rollout stays on-device in
        normalised space (``StandardScaler`` is linear, so denorm-then-renorm
        at every step is wasted work) and denormalises once at the end.

        For ``LSTMNN`` / ``RNN``, the hidden state is a local variable scoped
        to this call — ``self._hidden`` is untouched, and each of the B
        members carries its own independent slice of the hidden tensor across
        rollout steps. Permuting the batch order in gives the same outputs
        permuted.

        Parameters
        ----------
        state_histories : np.ndarray
            Shape ``(B, prev_time_steps, N)`` or ``(B, prev_time_steps * N)``,
            physical space, oldest-first.
        num_steps : int
            Number of future steps to predict.

        Returns
        -------
        np.ndarray of shape ``(B, num_steps, N)``.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        out = self.batch_rollout_tensor(state_histories, num_steps)
        return out.detach().cpu().numpy().astype(np.float64)

    def batch_rollout_tensor(self, state_histories, num_steps):
        """
        Batched autoregressive rollout; returns a GPU ``torch.Tensor`` in physical space.

        Same semantics as :meth:`batch_rollout`, but the result stays on
        ``self.device`` for zero-copy handoff to CuPy via DLPack.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        out_norm = self._batch_rollout_normalized_tensor(state_histories, num_steps)
        return self._denorm_tensor(out_norm)

    def batch_rollout_cupy(self, state_histories, num_steps):
        """
        Batched rollout returning a CuPy array on the default CUDA device (DLPack bridge).
        """
        import cupy as cp
        out = self.batch_rollout_tensor(state_histories, num_steps)
        if not out.is_cuda:
            raise RuntimeError("batch_rollout_cupy requires a CUDA SurrogateModel device.")
        return cp.from_dlpack(torch.utils.dlpack.to_dlpack(out.contiguous()))

    def _batch_rollout_normalized_tensor(self, state_histories, num_steps):
        """Rollout in normalised space; returns ``(B, num_steps, N)`` on device."""
        window = self._prepare_batch_tensor(state_histories)
        hidden = None
        preds = []

        with torch.inference_mode():
            for _ in range(num_steps):
                pred_norm, hidden = self._forward_normalized(window, hidden=hidden)
                preds.append(pred_norm)
                window = torch.cat([window[:, 1:, :], pred_norm.unsqueeze(1)], dim=1)

        return torch.stack(preds, dim=1)

    def _prepare_batch(self, state_histories):
        """Alias for :meth:`_prepare_batch_tensor` (backward compatibility)."""
        return self._prepare_batch_tensor(state_histories)

    def _prepare_batch_tensor(self, state_histories):
        """
        Validate, reshape, normalise, and move a batch of histories to device.

        Accepts NumPy (host), ``torch.Tensor``, or CuPy (GPU) arrays in **physical**
        space. Returns a tensor of shape ``(B, prev_time_steps, N)`` in normalised
        space on ``self.device``.
        """
        if isinstance(state_histories, torch.Tensor):
            arr_t = self._coerce_batch_tensor(state_histories)
        elif _is_cupy_array(state_histories):
            dtype = next(self.model.parameters()).dtype
            arr_t = torch.as_tensor(
                state_histories, device=self.device, dtype=dtype)
            arr_t = self._coerce_batch_tensor(arr_t)
        else:
            arr = np.asarray(state_histories, dtype=np.float64)
            batch_shape = self._parse_batch_shape(arr.shape, arr.ndim)
            if batch_shape[0] < 1:
                raise ValueError("Batch size B must be >= 1.")
            if arr.ndim == 2:
                arr = arr.reshape(batch_shape[0], self.prev_time_steps, self.input_size)
            dtype = next(self.model.parameters()).dtype
            arr_t = torch.as_tensor(arr, dtype=dtype, device=self.device)
        mean, std = self._norm_stats_tensors()
        return (arr_t - mean) / std

    def _coerce_batch_tensor(self, arr_t):
        """Reshape / validate a batch tensor already on device (physical space)."""
        if arr_t.device != self.device:
            arr_t = arr_t.to(self.device)
        dtype = next(self.model.parameters()).dtype
        if arr_t.dtype != dtype:
            arr_t = arr_t.to(dtype=dtype)
        if arr_t.ndim == 2:
            expected = self.prev_time_steps * self.input_size
            if arr_t.shape[1] != expected:
                raise ValueError(
                    f"Flat batch must have shape (B, {expected}), got {arr_t.shape}."
                )
            arr_t = arr_t.reshape(arr_t.shape[0], self.prev_time_steps, self.input_size)
        elif arr_t.ndim == 3:
            if (arr_t.shape[1] != self.prev_time_steps
                    or arr_t.shape[2] != self.input_size):
                raise ValueError(
                    f"Batch must have shape (B, {self.prev_time_steps}, {self.input_size}), "
                    f"got {arr_t.shape}."
                )
        else:
            raise ValueError(
                f"state_histories must be 2D or 3D; got ndim={arr_t.ndim}."
            )
        if arr_t.shape[0] < 1:
            raise ValueError("Batch size B must be >= 1.")
        return arr_t

    def _parse_batch_shape(self, shape, ndim):
        """Validate NumPy batch shape; return ``(B, prev_time_steps, N)`` dimensions."""
        if ndim == 2:
            if shape[1] != self.prev_time_steps * self.input_size:
                raise ValueError(
                    f"Flat batch must have shape (B, {self.prev_time_steps * self.input_size}), "
                    f"got {shape}."
                )
            return (shape[0], self.prev_time_steps, self.input_size)
        if ndim == 3:
            if shape[1] != self.prev_time_steps or shape[2] != self.input_size:
                raise ValueError(
                    f"Batch must have shape (B, {self.prev_time_steps}, {self.input_size}), "
                    f"got {shape}."
                )
            return shape
        raise ValueError(
            f"state_histories must be 2D (B, prev_time_steps * N) or "
            f"3D (B, prev_time_steps, N); got ndim={ndim}."
        )

    def _norm_stats_tensors(self):
        """Lazy device tensors for train_mean / train_std."""
        if self._train_mean_t is None:
            dtype = next(self.model.parameters()).dtype
            self._train_mean_t = torch.as_tensor(
                self.train_mean, dtype=dtype, device=self.device)
            self._train_std_t = torch.as_tensor(
                self.train_std, dtype=dtype, device=self.device)
        return self._train_mean_t, self._train_std_t

    def _denorm_tensor(self, x_norm):
        """Map normalised tensor(s) to physical space on device."""
        mean, std = self._norm_stats_tensors()
        return x_norm * std + mean

    def _forward_normalized(self, window, hidden):
        """
        Single forward pass over a batch already in normalised space on-device.

        Parameters
        ----------
        window : torch.Tensor
            Shape ``(B, prev_time_steps, N)``, normalised.
        hidden : recurrent hidden state or None.

        Returns
        -------
        (pred_normalised, hidden_next) — ``pred_normalised`` is
        ``(B, N)`` on-device; ``hidden_next`` is ``None`` for feedforward.
        """
        B = window.shape[0]
        flat = window.reshape(B, -1)
        if isinstance(self.model, (LSTMNN, RNN)):
            pred, hidden_next = self.model(flat, hidden)
            return pred, hidden_next
        pred = self.model(flat)
        return pred, None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_files(pth_path, yml_path):
        """Load checkpoint and YAML sidecar, returning (checkpoint_dict, yml_dict)."""
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

        # Auto-discover YAML sidecar
        if yml_path is None:
            base = pth_path.replace('.pth', '')
            # Strip _best_model suffix if present
            if base.endswith('_best_model'):
                base = base[:-len('_best_model')]
            candidate = base + '.yml'
            if os.path.isfile(candidate):
                yml_path = candidate

        yml_meta = {}
        if yml_path is not None:
            with open(yml_path, 'r') as f:
                raw = yaml.safe_load(f) or {}
            # Flatten: pull architecture sub-dict fields into top level
            arch_from_yml = raw.get('architecture', {})
            yml_meta = {**raw, **arch_from_yml}

        return checkpoint, yml_meta

    def _build_model(self, meta, yml_meta):
        """Instantiate the correct nn.Module from merged metadata."""
        model_type = meta['model_type']
        input_size = meta['input_size']
        prev_time_steps = meta['prev_time_steps']
        output_size = input_size
        hidden_layers = meta['hidden_layers']

        if model_type in ('DenseNN', 'ResDenseNN'):
            act_name = meta.get('hidden_activation',
                                yml_meta.get('hidden_activation', 'ReLU'))
            hidden_act = _ACTIVATION_MAP.get(act_name, nn.ReLU)
            cls = DenseNN if model_type == 'DenseNN' else ResDenseNN
            return cls(input_size, prev_time_steps, output_size,
                       hidden_layers, hidden_act, None)

        elif model_type == 'LSTMNN':
            hidden_size = hidden_layers[0]
            num_layers = int(meta.get('num_layers', yml_meta.get('num_layers', 1)))
            return LSTMNN(input_size, prev_time_steps, output_size,
                          hidden_size, num_layers)

        elif model_type == 'RNN':
            hidden_size = hidden_layers[0]
            num_layers = int(meta.get('num_layers', yml_meta.get('num_layers', 1)))
            nonlinearity = meta.get('rnn_nonlinearity',
                                    yml_meta.get('rnn_nonlinearity', 'tanh'))
            return RNN(input_size, prev_time_steps, output_size,
                       hidden_size, num_layers, nonlinearity)

        else:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                "Expected 'DenseNN', 'ResDenseNN', 'LSTMNN', or 'RNN'."
            )

    @staticmethod
    def _resolve_array(primary, fallback):
        """Return whichever source is available as a numpy array."""
        src = primary if primary is not None else fallback
        if src is None:
            raise ValueError(
                "Normalisation statistics (train_mean / train_std) not found in "
                "checkpoint or YAML sidecar."
            )
        return np.asarray(src, dtype=np.float64)
