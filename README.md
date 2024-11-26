# Simple Fourier transform

## Installation

```bash
git clone https://github.com/GianlucaP106/simple-fourier-transform
```

## Run Modes

```bash
python fft.py -i ./assets/moonlanding.png -m 1|2|3|4
```

#### Modes

1) Displays a log scale plot of the fourier transform of the provided image.
2) Displays a denoised (using fourier transform) version of the provided image.
3) Displays a the provided image compressed at various compression levels.
4) Displays a runtime plot showing the difference between naive dft2 (2D) and fft2 (2d).
