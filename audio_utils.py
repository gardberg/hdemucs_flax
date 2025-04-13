from jax import Array
import jax.scipy.signal as jsig
import jax.numpy as jnp
from scipy.signal import get_window

# corresponds to pytorch model's '_spectro'
def signal_to_spectrogram(x: Array, n_fft: int = 512, hop_length: int = 128, pad: int = 0) -> Array:
    """
    Calculate spectrogram of audio signal using JAX STFT

    Args:
        x: (batch_size, channels, time_steps)
    Returns:
        (batch_size, channels, freqs, time_steps)
    """
    length = x.shape[-1]
    remaining_dims = x.shape[:-1]
    x = x.reshape(-1, length)
    
    nperseg = n_fft * (1 + pad)  # equivalent to win_length in torch
    noverlap = nperseg - hop_length

    # First apply reflection padding ourselves since JAX STFT doesn't support it
    pad_size = nperseg // 2
    x = jnp.pad(x, ((0, 0), (pad_size, pad_size)), mode='reflect')

    # Compute STFT
    _, _, z = jsig.stft(
        x,
        fs=1.0,
        window='hann',  
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        boundary=None,
        padded=False,
        return_onesided=True,
    )

    z = z * jnp.sqrt(nperseg) * 0.5 # match pytorch normalization

    # Reshape output back to match input dimensions
    freqs = z.shape[1]
    frames = z.shape[2]
    return z.reshape(*remaining_dims, freqs, frames)


# corresponds to pytorch model's '_ispectro'
def spectrogram_to_signal(z: Array, hop_length: int = 128, length: int = 0, pad: int = 0) -> Array:
    """
    Convert complex spectrogram back to audio signal using JAX inverse STFT.
    Matches the behavior of torchaudio.models._hdemucs._ispectro.

    Args:
        z: (batch_size, channels, freqs, time_steps) of complex type
        hop_length: hop length for the STFT
        length: desired output length. If 0, use the length determined by istft.
        pad: padding factor applied when creating the spectrogram
    Returns:
        (batch_size, channels, time_steps)
    """
    remaining_dims = z.shape[:-2]
    freqs = z.shape[-2]
    frames = z.shape[-1]
    # Calculate combined batch/channel dimension safely
    batch_chan_dim = z.size // (freqs * frames) if freqs * frames > 0 else 0
    if batch_chan_dim == 0 and z.size > 0:
         raise ValueError("Invalid input shape for spectrogram.")
    elif batch_chan_dim > 0:
         z = z.reshape(batch_chan_dim, freqs, frames)
    # Handle empty input case
    else: # z.size == 0
        target_len = length if length > 0 else 0
        return jnp.zeros((*remaining_dims, target_len), dtype=jnp.float32)


    n_fft_istft = 2 * freqs - 2
    nperseg_istft = n_fft_istft // (1 + pad)
    noverlap = nperseg_istft - hop_length

    nperseg_fwd = nperseg_istft * (1 + pad)

    z = z / (jnp.sqrt(nperseg_fwd) * 0.5 + 1e-8)

    _, x = jsig.istft(
        z,
        fs=1.0,
        window='hann',
        nperseg=nperseg_istft,
        noverlap=noverlap,
        nfft=n_fft_istft,
        input_onesided=True,
        boundary=True, # Corresponds to center=True in torch.istft
        time_axis=-1,
        freq_axis=-2
    )

    # Trim or pad to target length if specified and different from istft output length
    istft_out_length = x.shape[-1]
    if length > 0 and length != istft_out_length:
        if istft_out_length > length:
            x = x[..., :length]
        else: # istft_out_length < length
            padding_amount = length - istft_out_length
            # Pad axis corresponding to time steps
            pad_widths = [(0, 0)] * (x.ndim - 1) + [(0, padding_amount)]
            x = jnp.pad(x, pad_widths)

    x = x.reshape(*remaining_dims, x.shape[-1])

    return x


def complex_spec_to_real(z: Array) -> Array:
    """
    Reshapes complex spectrogram to real spectrogram with double the channels

    Args:
        z: (batch_size, channels, freqs, time_steps) of dtype complex64
    Returns:
        (batch_size, channels * 2, freqs, time_steps) of dtype float32
    """
    B, C, F, T = z.shape
    m = jnp.stack([jnp.real(z), jnp.imag(z)], axis=-1)
    m = jnp.transpose(m, (0, 1, 4, 2, 3))
    return m.reshape(B, C * 2, F, T)

def real_spec_to_complex(m: Array) -> Array:
    """
    Reshapes real spectrogram to complex spectrogram

    Args:
        m: (batch_size, extra_dim, channels, freqs, time_steps) of dtype float32
    Returns:
        (batch_size, extra_dim, channels / 2, freqs, time_steps) of dtype complex64
    """
    B, S, C, F, T = m.shape
    out = m.reshape(B, S, -1, 2, F, T)
    out = jnp.transpose(out, (0, 1, 2, 4, 5, 3))
    real = out[..., 0]
    imag = out[..., 1]
    return real + 1j * imag