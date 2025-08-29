# ImageFX API - Python Implementation

Unofficial reverse engineered API for Google's ImageFX service.

## Files

- `imagefx.py` - Core API module
- `imagefx_streamlit.py` - Streamlit web interface
- `imagefx_cli.py` - Command line interface
- `cookie_parser.py` - Cookie file parser utility
- `requirements.txt` - Python dependencies
- `sample_prompts.txt` - Example prompts
- `cookies.txt` - Sample cookie file

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit app:
```bash
streamlit run imagefx_streamlit.py
```

3. Or use CLI:
```bash
python imagefx_cli.py --help
```

## Features

- ✅ Cookie file authentication
- ✅ Real-time image generation
- ✅ Multiple prompt support
- ✅ IMAGEN_3 aspect ratio models
- ✅ Download all images (ZIP)
- ✅ Failed prompt tracking
- ✅ Session state persistence
