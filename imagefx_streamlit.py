import streamlit as st
import requests
import base64
import io
from PIL import Image
import json
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from cookie_parser import CookieParser
from pathlib import Path
import tempfile
import zipfile
import hashlib
import time
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ImageFX API - Streamlit",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .prompt-input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        font-family: monospace;
        min-height: 120px;
    }
    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    .download-all-btn {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        margin: 20px 0;
    }
    .model-info {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    .failed-prompt-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 15px 0;
    }
    .real-time-container {
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8fff9;
    }
</style>
""", unsafe_allow_html=True)

# Data classes for type safety
@dataclass
class Credentials:
    cookie: Optional[str] = None
    authorization_key: Optional[str] = None
    cookie_file: Optional[str] = None

@dataclass
class Prompt:
    prompt: str
    count: int = 4
    seed: Optional[int] = None
    model: str = "IMAGEN_4"
    aspect_ratio: str = "IMAGE_ASPECT_RATIO_LANDSCAPE"

@dataclass
class GeneratedImage:
    encoded_image: str
    seed: int
    media_generation_id: str
    is_mask_edited_image: bool
    prompt: str
    model_name_type: str
    workflow_id: str
    fingerprint_log_record_id: str

def get_model_aspect_ratios(model: str) -> tuple:
    """
    Get the correct aspect ratios and model names based on the selected model
    Returns: (aspect_ratios_list, actual_model_name, aspect_ratio_value)
    """
    if model == "IMAGEN_3":
        # IMAGEN_3 has separate models for different aspect ratios
        aspect_ratios = [
            ("IMAGEN_3_LANDSCAPE", "Landscape (16:9) - IMAGEN_3_LANDSCAPE"),
            ("IMAGEN_3_PORTRAIT", "Portrait (9:16) - IMAGEN_3_PORTRAIT"),
            ("IMAGEN_3_PORTRAIT_THREE_FOUR", "Portrait (3:4) - IMAGEN_3_PORTRAIT_THREE_FOUR"),
            ("IMAGEN_3_LANDSCAPE_FOUR_THREE", "Landscape (4:3) - IMAGEN_3_LANDSCAPE_FOUR_THREE"),
        ]
        return aspect_ratios, "IMAGEN_3", "IMAGE_ASPECT_RATIO_UNSPECIFIED"
    
    elif model == "IMAGEN_2":
        # IMAGEN_2 has landscape variant
        aspect_ratios = [
            ("IMAGE_ASPECT_RATIO_LANDSCAPE", "Landscape (16:9)"),
            ("IMAGE_ASPECT_RATIO_SQUARE", "Square (1:1)"),
            ("IMAGE_ASPECT_RATIO_PORTRAIT", "Portrait (9:16)"),
            ("IMAGE_ASPECT_RATIO_UNSPECIFIED", "Unspecified (Let model decide)"),
        ]
        return aspect_ratios, "IMAGEN_2", "IMAGE_ASPECT_RATIO_LANDSCAPE"
    
    else:
        # IMAGEN_4, IMAGEN_3_1, IMAGEN_3_5 use standard aspect ratios
        aspect_ratios = [
            ("IMAGE_ASPECT_RATIO_LANDSCAPE", "Landscape (16:9)"),
            ("IMAGE_ASPECT_RATIO_SQUARE", "Square (1:1)"),
            ("IMAGE_ASPECT_RATIO_PORTRAIT", "Portrait (9:16)"),
            ("IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE", "Landscape (4:3) - Explicit"),
            ("IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR", "Portrait (3:4) - Explicit"),
            ("IMAGE_ASPECT_RATIO_UNSPECIFIED", "Unspecified (Let model decide)"),
        ]
        return aspect_ratios, model, "IMAGE_ASPECT_RATIO_LANDSCAPE"

class ImageFX:
    """Python implementation of the ImageFX API"""
    
    def __init__(self, credentials: Credentials):
        if not credentials.authorization_key and not credentials.cookie and not credentials.cookie_file:
            raise ValueError("Authorization token, Cookie, or Cookie file must be provided.")
        
        self.credentials = credentials
        
        # If cookie file is provided, parse it first
        if self.credentials.cookie_file:
            self._load_cookies_from_file()
        
        # Validate credentials after loading from file
        if not self.credentials.authorization_key and not self.credentials.cookie:
            raise ValueError("No valid authentication credentials found.")
        
        if self.credentials.cookie and len(self.credentials.cookie) < 70:
            raise ValueError("Invalid cookie was provided.")
        
        # Add the missing header
        if self.credentials.cookie and not self.credentials.cookie.startswith("__Secure-next-auth.session-token="):
            self.credentials.cookie = "__Secure-next-auth.session-token=" + self.credentials.cookie
    
    def _load_cookies_from_file(self):
        """Load cookies from Netscape cookie file"""
        try:
            if not Path(self.credentials.cookie_file).exists():
                raise FileNotFoundError(f"Cookie file not found: {self.credentials.cookie_file}")
            
            # Parse cookie file
            auth_data = CookieParser.get_auth_credentials(self.credentials.cookie_file)
            
            if auth_data['session_token']:
                self.credentials.cookie = auth_data['session_token']
                st.success(f"✅ Loaded session token from cookie file: {self.credentials.cookie_file}")
            else:
                raise ValueError("No session token found in cookie file")
                
        except Exception as e:
            raise ValueError(f"Failed to load cookies from file: {e}")
    
    def _make_request(self, url: str, method: str = "GET", headers: Optional[Dict] = None, 
                     body: Optional[str] = None) -> Dict[str, Any]:
        """Make HTTP request with proper headers"""
        default_headers = {
            "Origin": "https://labs.google",
            "Referer": "https://labs.google"
        }
        
        if headers:
            default_headers.update(headers)
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=default_headers)
            elif method.upper() == "POST":
                response = requests.post(url, headers=default_headers, data=body)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return {"Ok": response.text}
            
        except requests.exceptions.RequestException as e:
            return {"Err": str(e)}
    
    def check_token(self) -> Dict[str, Any]:
        """Check and validate authentication token"""
        if not self.credentials.authorization_key and not self.credentials.cookie:
            return {"Err": "Authorization token and Cookie both are missing."}
        
        if self.credentials.cookie and not self.credentials.authorization_key:
            # Get auth token internally
            result = self.get_auth_token(mutate=True)
            if "Err" in result:
                return result
            
            # Verify we now have the auth token
            if not self.credentials.authorization_key:
                return {"Err": "Failed to obtain authorization token from cookie"}
        
        return {"Ok": True}
    
    def get_auth_token(self, mutate: bool = False) -> Dict[str, Any]:
        """Get authentication token from cookie"""
        if not self.credentials.cookie:
            return {"Err": "Cookie is required for generating auth token."}
        
        result = self._make_request(
            url="https://labs.google/fx/api/auth/session",
            method="GET",
            headers={"Cookie": self.credentials.cookie}
        )
        
        if "Err" in result:
            return result
        
        try:
            parsed_resp = json.loads(result["Ok"])
            if "access_token" not in parsed_resp:
                return {"Err": f"Access token is missing from response: {result['Ok']}"}
            
            if mutate:
                self.credentials.authorization_key = parsed_resp["access_token"]
            
            return {"Ok": parsed_resp["access_token"]}
            
        except json.JSONDecodeError as e:
            return {"Err": f"Failed to parse response: {result['Ok']}"}
    
    def generate_image(self, prompt: Prompt) -> Dict[str, Any]:
        """Generate image from provided prompt"""
        token_res = self.check_token()
        if "Err" in token_res:
            return token_res
        
        # Handle IMAGEN_3 special case - use the aspect ratio as the model
        if prompt.model == "IMAGEN_3":
            # For IMAGEN_3, the aspect_ratio field contains the actual model name
            actual_model = prompt.aspect_ratio
            aspect_ratio = "IMAGE_ASPECT_RATIO_UNSPECIFIED"
        else:
            # For other models, use standard aspect ratios
            actual_model = prompt.model
            aspect_ratio = prompt.aspect_ratio
            
            # IMAGEN_3_5 is behind the scene IMAGEN_4
            if actual_model == "IMAGEN_4":
                actual_model = "IMAGEN_3_5"
        
        request_body = {
            "userInput": {
                "candidatesCount": prompt.count or 4,
                "prompts": [prompt.prompt],
                "seed": prompt.seed or 0,
            },
            "aspectRatio": aspect_ratio,
            "modelInput": {"modelNameType": actual_model},
            "clientContext": {"sessionId": ";1740658431200", "tool": "IMAGE_FX"},
        }
        
        result = self._make_request(
            url="https://aisandbox-pa.googleapis.com/v1:runImageFx",
            method="POST",
            headers={"Authorization": f"Bearer {self.credentials.authorization_key}"},
            body=json.dumps(request_body)
        )
        
        if "Err" in result:
            return result
        
        try:
            # Handle potential JSON truncation
            response_text = result["Ok"]
            if response_text.endswith("..."):
                st.warning("⚠️ Response appears to be truncated. Trying to parse what we have...")
            
            parsed_res = json.loads(response_text)
            
            # Check if we have the expected structure
            if "imagePanels" not in parsed_res:
                return {"Err": f"Unexpected response structure: {response_text[:200]}..."}
            
            if not parsed_res["imagePanels"]:
                return {"Err": "No image panels in response"}
            
            images = parsed_res["imagePanels"][0].get("generatedImages", [])
            
            if not isinstance(images, list):
                return {"Err": f"Invalid response received: {response_text[:200]}..."}
            
            if not images:
                return {"Err": "No images generated"}
            
            # Convert to GeneratedImage objects
            generated_images = []
            for img in images:
                try:
                    generated_images.append(GeneratedImage(
                        encoded_image=img["encodedImage"],
                        seed=img.get("seed", 0),
                        media_generation_id=img.get("mediaGenerationId", ""),
                        is_mask_edited_image=img.get("isMaskEditedImage", False),
                        prompt=img.get("prompt", prompt.prompt),
                        model_name_type=img.get("modelNameType", actual_model),
                        workflow_id=img.get("workflowId", ""),
                        fingerprint_log_record_id=img.get("fingerprintLogRecordId", "")
                    ))
                except KeyError as e:
                    st.warning(f"⚠️ Some image data missing: {e}")
                    continue
            
            if not generated_images:
                return {"Err": "Failed to parse any images from response"}
            
            return {"Ok": generated_images}
            
        except json.JSONDecodeError as e:
            return {"Err": f"Failed to parse JSON: {response_text[:200]}..."}
        except Exception as e:
            return {"Err": f"Unexpected error: {str(e)}"}

def save_image(image_data: str, filename: str) -> bool:
    """Save base64 image data to file"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Save to file
        with open(filename, "wb") as f:
            f.write(image_bytes)
        
        return True
    except Exception as e:
        st.error(f"Failed to save image: {e}")
        return False

def create_zip_file(images: List[GeneratedImage], title: str) -> bytes:
    """Create a ZIP file containing all images"""
    try:
        if not images:
            st.warning("⚠️ No images to add to ZIP file")
            return None
        
        if not title or not title.strip():
            st.error("❌ Invalid title provided for ZIP file")
            return None
            
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            successful_images = 0
            for i, image in enumerate(images):
                try:
                    # Validate image data
                    if not hasattr(image, 'encoded_image') or not image.encoded_image:
                        st.warning(f"⚠️ Image {i+1} has no encoded data, skipping...")
                        continue
                    
                    # Create filename: exact title + number
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_title = safe_title.replace(' ', '_')
                    
                    # Limit title length for safe filenames
                    if len(safe_title) > 50:
                        safe_title = safe_title[:50]
                    
                    filename = f"{safe_title}_{i + 1}.png"
                    
                    # Add image to ZIP
                    image_bytes = base64.b64decode(image.encoded_image)
                    zip_file.writestr(filename, image_bytes)
                    successful_images += 1
                except Exception as img_error:
                    st.warning(f"⚠️ Failed to add image {i+1} to ZIP: {img_error}")
                    continue
        
        if successful_images == 0:
            st.error("❌ No images were successfully added to ZIP file")
            return None
            
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Failed to create ZIP file: {e}")
        return None

def create_batch_zip_file(images: List[GeneratedImage], project_titles: List[str]) -> bytes:
    """Create a ZIP file containing all images organized by project"""
    try:
        if not images:
            st.warning("⚠️ No images to add to ZIP file")
            return None
        
        if not project_titles or not isinstance(project_titles, list):
            st.error("❌ Invalid project titles provided for batch ZIP file")
            return None
            
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            successful_images = 0
            for i, image in enumerate(images):
                try:
                    # Validate image data
                    if not hasattr(image, 'encoded_image') or not image.encoded_image:
                        st.warning(f"⚠️ Image {i+1} has no encoded data, skipping...")
                        continue
                    
                    # Get project title from image if available
                    project_title = getattr(image, 'project_title', project_titles[0] if project_titles else 'unknown')
                    
                    # Validate project title
                    if not project_title or not project_title.strip():
                        project_title = f"project_{i+1}"
                    
                    # Create filename: project_title + number
                    safe_title = "".join(c for c in project_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_title = safe_title.replace(' ', '_')
                    
                    # Limit title length for safe filenames
                    if len(safe_title) > 50:
                        safe_title = safe_title[:50]
                    
                    filename = f"{safe_title}_{i + 1}.png"
                    
                    # Add image to ZIP
                    image_bytes = base64.b64decode(image.encoded_image)
                    zip_file.writestr(filename, image_bytes)
                    successful_images += 1
                except Exception as img_error:
                    st.warning(f"⚠️ Failed to add image {i+1} to ZIP: {img_error}")
                    continue
        
        if successful_images == 0:
            st.error("❌ No images were successfully added to batch ZIP file")
            return None
            
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Failed to create batch ZIP file: {e}")
        return None

def reset_all_session_state():
    """Completely reset all session state variables"""
    # Clear all generation-related state
    st.session_state.failed_prompts = []
    st.session_state.generation_complete = False
    st.session_state.all_generated_images = []
    st.session_state.current_title = ""
    st.session_state.show_download_button = False
    st.session_state.generating = False
    
    # Clear timing information
    if 'generation_start_time' in st.session_state:
        del st.session_state.generation_start_time
    if 'generation_end_time' in st.session_state:
        del st.session_state.generation_end_time
    if 'generation_start_datetime' in st.session_state:
        del st.session_state.generation_start_datetime
    if 'generation_end_datetime' in st.session_state:
        del st.session_state.generation_end_datetime
    if 'total_generation_time' in st.session_state:
        del st.session_state.total_generation_time
    
    # Clear project data
    if 'project_titles' in st.session_state:
        del st.session_state.project_titles
    if 'project_prompts' in st.session_state:
        del st.session_state.project_prompts
    
    # Clear any other session state that might cause issues
    if 'page_refresh' in st.session_state:
        del st.session_state.page_refresh

def display_image(image_data: str, caption: str, title: str, index: int, prompt_text: str):
    """Display image in Streamlit with title-based naming"""
    try:
        # Validate input parameters
        if not image_data or not isinstance(image_data, str):
            st.error("❌ Invalid image data provided")
            return
            
        if not title or not isinstance(title, str):
            title = f"image_{index + 1}"
            
        if not prompt_text or not isinstance(prompt_text, str):
            prompt_text = "No prompt available"
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as decode_error:
            st.error(f"❌ Failed to decode image data: {decode_error}")
            return
        
        # Create filename based on exact title
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        # Limit title length for safe filenames
        if len(safe_title) > 50:
            safe_title = safe_title[:50]
            
        filename = f"{safe_title}_{index + 1}.png"
        
        # Display image in container with proper sizing
        with st.container():
            st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
            
            # Show prompt info (truncate if too long)
            display_prompt = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
            st.markdown(f"**Prompt:** {display_prompt}")
            
            # Display image with controlled size
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption=caption, width='content')
            
            # Add download button with unique key to prevent clearing
            # Use hash of prompt text to ensure uniqueness even with similar prompts
            try:
                prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
                download_key = f"download_{index}_{prompt_hash}"
            except Exception as hash_error:
                # Fallback to simple key if hashing fails
                download_key = f"download_{index}_{index}"
            
            st.download_button(
                label=f"📥 Download {filename}",
                data=image_bytes,
                file_name=filename,
                mime="image/png",
                key=download_key,
                width='content'
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Failed to display image: {e}")
        # Show a placeholder for failed images
        st.markdown(f"""
        <div class="error-box">
        <strong>Image Display Failed:</strong><br>
        Error: {str(e)[:100]}...<br>
        Index: {index}<br>
        Title: {title[:50] if title else 'Unknown'}
        </div>
        """, unsafe_allow_html=True)

def main():
    # Check if we're in clearing state - if so, don't show anything
    if st.session_state.get('clearing_state', False):
        st.info("🔄 Clearing application state... Please wait.")
        return
    
    st.markdown('<h1 class="main-header">🎨 ImageFX API - Streamlit</h1>', unsafe_allow_html=True)
    st.markdown("Unofficial reverse engineered API for imageFX service provided by Google Labs")
    
    # Check if this is a fresh page load (no generation in progress)
    if 'page_refresh' not in st.session_state:
        st.session_state.page_refresh = True
        # Reset all generation-related flags on fresh page load
        reset_all_session_state()
    
    # Initialize session state for failed prompts and generation state
    # Use get() method for safer initialization
    st.session_state.setdefault('failed_prompts', [])
    st.session_state.setdefault('generation_complete', False)
    st.session_state.setdefault('all_generated_images', [])
    st.session_state.setdefault('current_title', "")
    st.session_state.setdefault('generating', False)
    st.session_state.setdefault('show_download_button', False)  # Control when to show download buttons
    
    # Ensure all_generated_images is always a list
    if not isinstance(st.session_state.get('all_generated_images'), list):
        st.session_state.all_generated_images = []
    
    # Ensure failed_prompts is always a list
    if not isinstance(st.session_state.get('failed_prompts'), list):
        st.session_state.failed_prompts = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔐 Authentication")
        
        auth_method = st.radio(
            "Choose authentication method:",
            ["Cookie File", "Cookie", "Auth Token"],
            help="Cookie file is easiest - just upload your cookies.txt file!"
        )
        
        cookie_file = None
        cookie = None
        auth_token = None
        
        if auth_method == "Cookie File":
            st.markdown("**Upload your cookies.txt file:**")
            uploaded_file = st.file_uploader(
                "Choose cookies.txt file",
                type=['txt'],
                help="Upload the cookies.txt file from your browser"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue().decode())
                    cookie_file = tmp_file.name
                
                # Parse and show cookie info
                try:
                    auth_data = CookieParser.get_auth_credentials(cookie_file)
                    if auth_data['session_token']:
                        st.success("✅ Cookie file loaded successfully!")
                        st.info(f"Session token found: {auth_data['session_token'][:30]}...")
                    else:
                        st.error("❌ No session token found in cookie file")
                except Exception as e:
                    st.error(f"❌ Error parsing cookie file: {e}")
        
        elif auth_method == "Cookie":
            cookie = st.text_input(
                "Cookie (__Secure-next-auth.session-token)",
                type="password",
                help="Extract from browser developer tools"
            )
        else:
            auth_token = st.text_input(
                "Authentication Token",
                type="password",
                help="Extract from browser console"
            )
        
        st.markdown("---")
        st.header("⚙️ Generation Settings")
        
        model = st.selectbox(
            "Model",
            ["IMAGEN_4", "IMAGEN_3", "IMAGEN_2", "IMAGEN_3_1", "IMAGEN_3_5"],
            help="AI model to use for generation"
        )
        
        # Get aspect ratios based on selected model
        aspect_ratios, actual_model, default_aspect = get_model_aspect_ratios(model)
        
        # Show model info
        if model == "IMAGEN_3":
            st.markdown(f"""
            <div class="model-info">
            <strong>ℹ️ IMAGEN_3 Special Note:</strong><br>
            IMAGEN_3 uses separate models for different aspect ratios.<br>
            The aspect ratio selection below will choose the correct IMAGEN_3 variant.
            </div>
            """, unsafe_allow_html=True)
        
        # Create aspect ratio options
        aspect_ratio_options = [ratio[0] for ratio in aspect_ratios]
        aspect_ratio_labels = [ratio[1] for ratio in aspect_ratios]
        
        # Create mapping for display
        aspect_ratio_mapping = dict(zip(aspect_ratio_labels, aspect_ratio_options))
        
        aspect_ratio_label = st.selectbox(
            "Aspect Ratio",
            aspect_ratio_labels,
            help="Output image dimensions and model variant"
        )
        
        # Get the actual aspect ratio value
        aspect_ratio = aspect_ratio_mapping[aspect_ratio_label]
        
        count = st.slider("Number of Images", 1, 10, 4, help="How many images to generate")
        
        seed = st.number_input(
            "Seed (Optional)",
            min_value=0,
            value=None,
            help="Same seed generates similar images. Leave empty for random."
        )
        
        # Show current configuration
        st.markdown("---")
        st.markdown("**🔧 Current Configuration:**")
        st.code(f"Model: {actual_model}\nAspect Ratio: {aspect_ratio}\nCount: {count}\nSeed: {seed or 'Random'}")
        
        # Help section
        with st.expander("ℹ️ How to get credentials?"):
            st.markdown("""
            **For Cookie Files (Easiest):**
            1. Open [labs.google](https://labs.google/fx/tools/image-fx)
            2. Make sure you're logged in
            3. Install a browser extension like "Cookie Editor" or "EditThisCookie"
            4. Export cookies as Netscape format (cookies.txt)
            5. Upload the file here!
            
            **For Cookies:**
            1. Open [labs.google](https://labs.google/fx/tools/image-fx)
            2. Press `Ctrl+Shift+I` to open console
            3. Click `Application` tab → `Cookies` → `https://labs.google`
            4. Copy value of `__Secure-next-auth.session-token`
            
            **For Auth Token:**
            1. Open [labs.google](https://labs.google/fx/tools/image-fx)
            2. Press `Ctrl+Shift+I` to open console
            3. Paste this code:
            ```js
            let script = document.querySelector("#__NEXT_DATA__");
            let obj = JSON.parse(script.textContent);
            let authToken = obj["props"]["pageProps"]["session"]["access_token"];
            window.prompt("Copy the auth token: ", authToken);
            ```
            4. Copy the token from the prompt box
            """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Image Generation")
        
        # Project mode selection
        project_mode = st.radio(
            "🎯 Project Mode:",
            ["Single Project", "Batch Projects"],
            help="Choose between single project or multiple batch projects"
        )
        
        if project_mode == "Single Project":
            # Single project title field
            title = st.text_input(
                "🎯 Project Title",
                placeholder="Enter a title for your image project (used for file naming)",
                help="This title will be used to name your downloaded images (max 100 characters)",
                max_chars=100
            )
            
            # Show single project limits
            st.markdown("""
            <div class="info-box">
            <strong>📊 Single Project Limits:</strong><br>
            • <strong>Maximum Prompts:</strong> 50<br>
            • <strong>Title Length:</strong> 100 characters<br>
            • <strong>Images per Prompt:</strong> 1-10 (based on your count setting)
            </div>
            """, unsafe_allow_html=True)
            
            if not title.strip():
                st.warning("⚠️ Please enter a project title for better file organization!")
            elif len(title) > 100:
                st.warning("⚠️ Title is very long. Consider using a shorter title for better file naming.")
            
            # Multiple prompts field for single project
            st.markdown("**📝 Prompts (one per line):**")
            st.info("💡 **Important:** You can enter up to 50 prompts for this project!")
            prompts_text = st.text_area(
                "Enter your prompts:",
                placeholder="A majestic purple cat with golden wings\n\nA cyberpunk city at night\n\nA wise old wizard in a forest",
                height=120,
                help="Enter multiple prompts, one per line. Empty lines are automatically ignored. Each prompt will generate images separately. Maximum 50 prompts per project."
            )
            
            # Show real-time prompt count for single project
            if prompts_text.strip():
                prompt_lines = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                prompt_count = len(prompt_lines)
                
                if prompt_count > 50:
                    st.error(f"❌ **Too many prompts:** {prompt_count}/50 (Maximum allowed)")
                elif prompt_count > 40:
                    st.warning(f"⚠️ **Prompt count:** {prompt_count}/50 (Getting close to limit)")
                elif prompt_count > 20:
                    st.info(f"📝 **Prompt count:** {prompt_count}/50")
                else:
                    st.success(f"✅ **Prompt count:** {prompt_count}/50")
            
            # Store in session state for single project
            st.session_state.project_titles = [title] if title.strip() else []
            st.session_state.project_prompts = [prompts_text] if prompts_text.strip() else []
            
        else:
            # Batch projects mode
            st.markdown("**📚 Batch Projects - Create multiple projects with separate prompts**")
            st.info("💡 **Important:** Each project can have up to 50 prompts. The limit is per project, not combined total!")
            
            # Show limits summary
            st.markdown("""
            <div class="info-box">
            <strong>📊 Batch Project Limits:</strong><br>
            • <strong>Maximum Projects:</strong> 10<br>
            • <strong>Maximum Prompts per Project:</strong> 50<br>
            • <strong>Total Prompts Across All Projects:</strong> Unlimited (as long as each project stays under 50)<br>
            • <strong>Example:</strong> You can have 10 projects with 50 prompts each = 500 total prompts
            </div>
            """, unsafe_allow_html=True)
            
            # Number of projects
            num_projects = st.number_input(
                "Number of Projects:",
                min_value=1,
                max_value=10,
                value=3,
                help="How many separate projects to create (max 10). Each project can have up to 50 prompts individually."
            )
            
            # Initialize batch projects in session state
            if 'batch_projects' not in st.session_state:
                st.session_state.batch_projects = []
            
            # Create project inputs
            project_titles = []
            project_prompts = []
            
            for i in range(num_projects):
                st.markdown(f"---")
                st.markdown(f"### 🎯 Project {i+1}")
                
                # Project title
                project_title = st.text_input(
                    f"Project {i+1} Title:",
                    placeholder=f"Project {i+1} Title",
                    help=f"Title for project {i+1} (max 100 characters)",
                    max_chars=100,
                    key=f"title_{i}"
                )
                
                if not project_title.strip():
                    st.warning(f"⚠️ Please enter a title for Project {i+1}!")
                elif len(project_title) > 100:
                    st.warning(f"⚠️ Project {i+1} title is very long. Consider using a shorter title.")
                
                # Project prompts
                project_prompt = st.text_area(
                    f"Project {i+1} Prompts (one per line):",
                    placeholder=f"Enter prompts for project {i+1}...",
                    height=100,
                    help=f"Enter multiple prompts for project {i+1}, one per line. Empty lines are ignored. Maximum 50 prompts per project.",
                    key=f"prompts_{i}"
                )
                
                # Show real-time prompt count for this project
                if project_prompt.strip():
                    prompt_lines = [p.strip() for p in project_prompt.split('\n') if p.strip()]
                    prompt_count = len(prompt_lines)
                    
                    if prompt_count > 50:
                        st.error(f"❌ **Too many prompts:** {prompt_count}/50 (Maximum allowed)")
                    elif prompt_count > 40:
                        st.warning(f"⚠️ **Prompt count:** {prompt_count}/50 (Getting close to limit)")
                    elif prompt_count > 20:
                        st.info(f"📝 **Prompt count:** {prompt_count}/50")
                    else:
                        st.success(f"✅ **Prompt count:** {prompt_count}/50")
                
                if project_title.strip() and project_prompt.strip():
                    project_titles.append(project_title)
                    project_prompts.append(project_prompt)
                else:
                    st.warning(f"⚠️ Project {i+1} needs both title and prompts!")
            
            # Store batch projects in session state
            st.session_state.project_titles = project_titles
            st.session_state.project_prompts = project_prompts
            
            # Show batch summary
            if project_titles and project_prompts:
                st.success(f"✅ Batch Projects Ready: {len(project_titles)} projects with prompts")
                st.markdown("**📋 Batch Summary:**")
                for i, (title, prompts) in enumerate(zip(project_titles, project_prompts)):
                    prompt_count = len([p for p in prompts.split('\n') if p.strip()])
                    status_icon = "✅" if prompt_count <= 50 else "❌"
                    limit_status = f" ({prompt_count}/50)" if prompt_count <= 50 else f" ({prompt_count}/50 - EXCEEDS LIMIT!)"
                    st.info(f"{status_icon} **Project {i+1}:** {title}{limit_status}")
            
            # For batch mode, we'll use the first project as the main title for display
            title = project_titles[0] if project_titles else ""
            prompts_text = project_prompts[0] if project_prompts else ""
        
        # Download All Images Button at the top (only show after new generation)
        # Only show download button if we just completed generation, not on page load
        if (st.session_state.generation_complete and 
            st.session_state.show_download_button and  # New flag to control when to show download
            st.session_state.all_generated_images and 
            len(st.session_state.all_generated_images) > 0 and  # Ensure there are actually images
            not st.session_state.generating and  # Don't show while generating
            hasattr(st.session_state, 'current_title') and 
            st.session_state.current_title and  # Only show if we have a valid title
            not st.session_state.get('clearing_state', False)):  # Don't show while clearing
            
            # Get the actual images list
            images_to_zip = st.session_state.all_generated_images
            
            st.markdown("---")
            st.markdown("### 📥 Download All Generated Images")
            
            # Create ZIP with project organization
            if project_mode == "Batch Projects" and hasattr(st.session_state, 'project_titles'):
                # For batch projects, organize by project
                zip_data = create_batch_zip_file(images_to_zip, st.session_state.project_titles)
                zip_filename = "batch_projects_all_images.zip"
            else:
                # For single project
                zip_data = create_zip_file(images_to_zip, title)
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')
                zip_filename = f"{safe_title}_all_images.zip"
            
            if zip_data:
                st.download_button(
                    label="📦 Download All Images (ZIP)",
                    data=zip_data,
                    file_name=zip_filename,
                    mime="application/zip",
                    width='stretch',
                    key="download_all_main"
                )
                st.info(f"📁 ZIP file contains {len(images_to_zip)} images")
                
                # Add a note about individual downloads
                st.info("💡 Individual download buttons are also available below each image")
        
        # Display previously generated images if generation is complete
        if (st.session_state.generation_complete and 
            st.session_state.show_download_button and  # Only show after new generation
            st.session_state.all_generated_images and 
            len(st.session_state.all_generated_images) > 0 and  # Ensure there are actually images
            not st.session_state.generating and  # Don't show while generating
            hasattr(st.session_state, 'current_title') and 
            st.session_state.current_title and  # Only show if we have a valid title
            not st.session_state.get('clearing_state', False)):  # Don't show while clearing
            
            st.markdown("---")
            st.markdown("### 🖼️ Generated Images")
            st.info("These images were just generated. You can download them individually or use the ZIP download above.")
            
            # Display all images
            try:
                for i, image in enumerate(st.session_state.all_generated_images):
                    try:
                        # Safely get prompt text
                        prompt_text = getattr(image, 'prompt', 'No prompt available')
                        if not prompt_text:
                            prompt_text = 'No prompt available'
                        
                        # Truncate prompt for display
                        display_prompt = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
                        st.markdown(f"#### Image {i+1} - '{display_prompt}'")
                        
                        # Safely get image data
                        if hasattr(image, 'encoded_image') and image.encoded_image:
                            display_image(image.encoded_image, f"Generated Image {i+1}", st.session_state.current_title, i, prompt_text)
                        else:
                            st.error(f"❌ Image {i+1} has no encoded data")
                    except Exception as img_error:
                        st.error(f"❌ Failed to display image {i+1}: {img_error}")
                        continue
            except Exception as display_error:
                st.error(f"❌ Failed to display images: {display_error}")
        

        
        # Generate button with loading state
        button_text = "🔄 Generating..." if st.session_state.generating else "🚀 Generate Images"
        if st.button(button_text, type="primary", width='stretch', disabled=st.session_state.generating):
            # Check if we have batch projects or single project
            if project_mode == "Batch Projects":
                if not st.session_state.project_titles or not st.session_state.project_prompts:
                    st.error("Please create at least one batch project with title and prompts!")
                    return
                
                # Validate all batch projects
                valid_projects = []
                for i, (project_title, project_prompts_text) in enumerate(zip(st.session_state.project_titles, st.session_state.project_prompts)):
                    if not project_title.strip() or not project_prompts_text.strip():
                        st.error(f"Project {i+1} is missing title or prompts!")
                        return
                    
                    # Parse prompts for this project
                    raw_prompts = project_prompts_text.split('\n')
                    prompts = [p.strip() for p in raw_prompts if p.strip()]
                    
                    if not prompts:
                        st.error(f"Project {i+1} has no valid prompts!")
                        return
                    
                    valid_projects.append({
                        'title': project_title,
                        'prompts': prompts,
                        'project_index': i + 1
                    })
                
                st.success(f"📝 Processing {len(valid_projects)} batch projects...")
                
            else:
                # Single project mode
                if not prompts_text.strip():
                    st.error("Please enter at least one prompt!")
                    return
                
                if not title.strip():
                    st.error("Please enter a project title!")
                    return
                
                # Parse prompts - ignore empty lines and raw data
                raw_prompts = prompts_text.split('\n')
                prompts = []
                
                for i, raw_prompt in enumerate(raw_prompts):
                    cleaned_prompt = raw_prompt.strip()
                    if cleaned_prompt:  # Only add non-empty prompts
                        prompts.append(cleaned_prompt)
                    else:
                        st.info(f"⚠️ Skipping empty line {i+1}")
                
                if not prompts:
                    st.error("No valid prompts found! Please enter at least one non-empty prompt.")
                    return
                
                valid_projects = [{
                    'title': title,
                    'prompts': prompts,
                    'project_index': 1
                }]
            
            # Validate authentication credentials
            if not cookie and not auth_token and not cookie_file:
                st.error("❌ Please provide authentication credentials!")
                return
            
            # Additional validation for cookie file
            if cookie_file and not Path(cookie_file).exists():
                st.error("❌ Cookie file not found or invalid!")
                return
            
            # Set generating state and start timing
            st.session_state.generating = True
            st.session_state.generation_start_time = time.time()
            st.session_state.generation_start_datetime = datetime.now()
            
            # Clear previous failed prompts
            st.session_state.failed_prompts = []
            
            # Show generation info
            st.info(f"🎨 Using Model: {actual_model} | Aspect Ratio: {aspect_ratio}")
            
            # Create containers for real-time display
            st.markdown("### 🖼️ Generated Images (Real-time)")
            
            # Add progress bar and timer for generation
            progress_bar = st.progress(0)
            status_text = st.empty()
            timer_text = st.empty()  # For real-time timer display
            
            # Initialize variables
            st.session_state.all_generated_images = []
            st.session_state.failed_prompts = []
            st.session_state.generation_complete = False
            st.session_state.current_title = valid_projects[0]['title'] if valid_projects else ""
            
            # Validate prompt count for each project individually (not combined)
            for i, project in enumerate(valid_projects):
                project_prompts_count = len(project['prompts'])
                if project_prompts_count > 50:
                    st.error(f"❌ Project {i+1} '{project['title']}' has too many prompts ({project_prompts_count}). Maximum allowed is 50 prompts per project.")
                    st.session_state.generating = False
                    return
                elif project_prompts_count > 20:
                    st.warning(f"⚠️ Project {i+1} '{project['title']}' has many prompts ({project_prompts_count}). This may take a while.")
            
            # Show total prompts info (for information only, not for limiting)
            total_prompts = sum(len(project['prompts']) for project in valid_projects)
            if total_prompts > 100:
                st.info(f"📊 Total prompts across all projects: {total_prompts} (This is informational only - each project is limited to 50 prompts)")
            
            # Process prompts in parallel batches for faster generation
            # Note: ThreadPoolExecutor warnings about missing ScriptRunContext are normal and can be ignored
            import concurrent.futures
            import threading
            
            # Create a lock for thread-safe operations
            lock = threading.Lock()
            
            def process_single_prompt(prompt_data):
                prompt_index, prompt_text = prompt_data
                
                try:
                    # Initialize ImageFX
                    credentials = Credentials(
                        cookie=cookie, 
                        authorization_key=auth_token,
                        cookie_file=cookie_file
                    )
                    imagefx = ImageFX(credentials)
                    
                    # Create prompt object
                    prompt_obj = Prompt(
                        prompt=prompt_text.strip(),
                        count=count,
                        seed=seed,
                        model=model,
                        aspect_ratio=aspect_ratio
                    )
                    
                    # Generate images
                    result = imagefx.generate_image(prompt_obj)
                    
                    if "Err" in result:
                        # Store error locally first, then update session state safely
                        error_data = {
                            "prompt": prompt_text,
                            "error": result['Err'],
                            "index": prompt_index + 1
                        }
                        return None, prompt_text, result['Err'], error_data
                    
                    # Add prompt info to images
                    for img in result['Ok']:
                        img.prompt = prompt_text  # Override with the actual prompt used
                    
                    return result['Ok'], prompt_text, None, None
                    
                except Exception as e:
                    # Store error locally first, then update session state safely
                    error_data = {
                        "prompt": prompt_text,
                        "error": str(e),
                        "index": prompt_index + 1
                    }
                    return None, prompt_text, str(e), error_data
            

            # Process all projects
            total_projects = len(valid_projects)
            project_progress = 0
            
            # Start real-time timer updates
            import threading
            
            def update_timer():
                try:
                    while st.session_state.get('generating', False):
                        if hasattr(st.session_state, 'generation_start_time'):
                            elapsed = time.time() - st.session_state.generation_start_time
                            if elapsed < 60:
                                timer_display = f"⏱️ **Elapsed Time:** {elapsed:.1f}s"
                            else:
                                minutes = int(elapsed // 60)
                                seconds = int(elapsed % 60)
                                timer_display = f"⏱️ **Elapsed Time:** {minutes}m {seconds}s"
                            timer_text.markdown(timer_display)
                        time.sleep(1)
                except Exception as e:
                    # Timer thread error - just log and continue
                    st.warning(f"⚠️ Timer update error: {e}")
            
            # Start timer in background thread
            timer_thread = threading.Thread(target=update_timer, daemon=True)
            timer_thread.start()
            
            for project_idx, project in enumerate(valid_projects):
                project_title = project['title']
                project_prompts = project['prompts']
                project_index = project['project_index']
                
                st.markdown(f"---")
                st.markdown(f"#### 🚀 Processing Project {project_index}: {project_title}")
                st.info(f"📝 Project '{project_title}' has {len(project_prompts)} prompts")
                
                # Update progress bar for project
                try:
                    project_progress = project_idx / total_projects
                    progress_bar.progress(project_progress)
                except Exception as progress_error:
                    st.warning(f"⚠️ Progress bar update failed: {progress_error}")
                
                # Show timing information in status
                try:
                    if hasattr(st.session_state, 'generation_start_time'):
                        elapsed_time = time.time() - st.session_state.generation_start_time
                        if elapsed_time < 60:
                            elapsed_display = f"{elapsed_time:.1f}s"
                        else:
                            minutes = int(elapsed_time // 60)
                            seconds = int(elapsed_time % 60)
                            elapsed_display = f"{minutes}m {seconds}s"
                        status_text.text(f"🔄 Processing Project {project_index}/{total_projects}: {project_title} | ⏱️ Elapsed: {elapsed_display}")
                    else:
                        status_text.text(f"🔄 Processing Project {project_index}/{total_projects}: {project_title}")
                except Exception as status_error:
                    st.warning(f"⚠️ Status update failed: {status_error}")
                
                # Process prompts for this project in batches
                batch_size = 3
                total_batches = (len(project_prompts) + batch_size - 1) // batch_size
                
                for batch_start in range(0, len(project_prompts), batch_size):
                    batch_end = min(batch_start + batch_size, len(project_prompts))
                    batch_prompts = list(enumerate(project_prompts[batch_start:batch_end], batch_start + 1))
                    current_batch = batch_start // batch_size + 1
                    
                    st.markdown(f"**Batch {current_batch}/{total_batches}:** Prompts {batch_start + 1}-{batch_end}")
                    
                    # Show timing for this batch
                    if hasattr(st.session_state, 'generation_start_time'):
                        elapsed_time = time.time() - st.session_state.generation_start_time
                        if elapsed_time < 60:
                            elapsed_display = f"{elapsed_time:.1f}s"
                        else:
                            minutes = int(elapsed_time // 60)
                            seconds = int(elapsed_time % 60)
                            elapsed_display = f"{minutes}m {seconds}s"
                        st.info(f"⏱️ **Batch {current_batch} Elapsed Time:** {elapsed_display}")
                    
                    # Process batch in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                        future_to_prompt = {executor.submit(process_single_prompt, prompt_data): prompt_data for prompt_data in batch_prompts}
                        
                        # Collect results safely
                        batch_results = []
                        
                        for future in concurrent.futures.as_completed(future_to_prompt):
                            prompt_index, prompt_text = future_to_prompt[future]
                            try:
                                result = future.result()
                                if len(result) == 4:  # New format with error_data
                                    images, prompt_text, error, error_data = result
                                else:  # Fallback for old format
                                    images, prompt_text, error = result
                                    error_data = None
                                
                                batch_results.append({
                                    'prompt_index': prompt_index,
                                    'prompt_text': prompt_text,
                                    'images': images,
                                    'error': error,
                                    'error_data': error_data,
                                    'project_title': project_title
                                })
                                
                            except Exception as e:
                                st.error(f"❌ Error processing prompt {prompt_index}: {str(e)}")
                                batch_results.append({
                                    'prompt_index': prompt_index,
                                    'prompt_text': prompt_text,
                                    'images': None,
                                    'error': str(e),
                                    'error_data': None,
                                    'project_title': project_title
                                })
                        
                        # Now safely update session state and display results
                        for result in batch_results:
                            prompt_index = result['prompt_index']
                            prompt_text = result['prompt_text']
                            images = result['images']
                            error = result['error']
                            error_data = result['error_data']
                            project_title = result['project_title']
                            
                            if error:
                                st.error(f"❌ Generation failed for Project '{project_title}' - Prompt {prompt_index}: {error}")
                                
                                # Safely add to failed prompts if we have error data
                                if error_data and hasattr(st.session_state, 'failed_prompts'):
                                    error_data['project_title'] = project_title
                                    st.session_state.failed_prompts.append(error_data)
                                    
                            elif images:
                                st.success(f"✅ Generated {len(images)} images for Project '{project_title}' - Prompt {prompt_index}")
                                
                                # Add project title to images
                                for img in images:
                                    img.project_title = project_title
                                
                                # Safely add to generated images
                                if hasattr(st.session_state, 'all_generated_images'):
                                    st.session_state.all_generated_images.extend(images)
                                
                                # Display images immediately (real-time)
                                st.markdown(f'<div class="real-time-container">', unsafe_allow_html=True)
                                st.markdown(f"**📸 Project '{project_title}' - Images for Prompt {prompt_index}:**")
                                
                                # Show timing for this batch
                                if hasattr(st.session_state, 'generation_start_time'):
                                    elapsed_time = time.time() - st.session_state.generation_start_time
                                    if elapsed_time < 60:
                                        elapsed_display = f"{elapsed_time:.1f}s"
                                    else:
                                        minutes = int(elapsed_time // 60)
                                        seconds = int(elapsed_time % 60)
                                        elapsed_display = f"{minutes}m {seconds}s"
                                    st.info(f"⏱️ **Elapsed Time:** {elapsed_display}")
                                
                                for i, image in enumerate(images):
                                    try:
                                        # Use a unique global index for each image across all projects
                                        global_image_index = len(st.session_state.all_generated_images) + i
                                        display_image(image.encoded_image, f"Generated Image {i+1}", project_title, global_image_index, prompt_text)
                                    except Exception as display_error:
                                        st.error(f"❌ Failed to display image {i+1}: {display_error}")
                                        continue
                                
                                st.markdown('</div>', unsafe_allow_html=True)
            
            # Mark generation as complete and reset generating state
            st.session_state.generation_complete = True
            st.session_state.generating = False
            st.session_state.show_download_button = True  # Enable download buttons only after generation
            
            # Calculate timing information
            st.session_state.generation_end_time = time.time()
            st.session_state.generation_end_datetime = datetime.now()
            st.session_state.total_generation_time = st.session_state.generation_end_time - st.session_state.generation_start_time
            
            # Update progress bar to 100%
            try:
                progress_bar.progress(1.0)
                status_text.text("✅ Generation Complete!")
                
                # Show final timer display
                if hasattr(st.session_state, 'total_generation_time'):
                    total_time = st.session_state.total_generation_time
                    if total_time < 60:
                        final_timer = f"⏱️ **Total Generation Time:** {total_time:.1f} seconds"
                    elif total_time < 3600:
                        minutes = int(total_time // 60)
                        seconds = int(total_time % 60)
                        final_timer = f"⏱️ **Total Generation Time:** {minutes}m {seconds}s"
                    else:
                        hours = int(total_time // 3600)
                        minutes = int((total_time % 3600) // 60)
                        final_timer = f"⏱️ **Total Generation Time:** {hours}h {minutes}m"
                    
                    timer_text.markdown(final_timer)
                    
            except Exception as e:
                st.warning(f"⚠️ Progress bar update failed: {e}")
                status_text.text("✅ Generation Complete!")
            
            # Final summary
            st.markdown("---")
            st.markdown("### 🎉 Generation Complete!")
            
            # Show timing information
            if hasattr(st.session_state, 'total_generation_time'):
                total_time = st.session_state.total_generation_time
                start_time = st.session_state.generation_start_datetime.strftime("%H:%M:%S")
                end_time = st.session_state.generation_end_datetime.strftime("%H:%M:%S")
                
                # Format time nicely
                if total_time < 60:
                    time_display = f"{total_time:.1f} seconds"
                elif total_time < 3600:
                    minutes = int(total_time // 60)
                    seconds = int(total_time % 60)
                    time_display = f"{minutes}m {seconds}s"
                else:
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    seconds = int(total_time % 60)
                    time_display = f"{hours}h {minutes}m {seconds}s"
                
                st.success(f"⏱️ **Generation Time:** {time_display}")
                st.info(f"🕐 **Started:** {start_time} | **Completed:** {end_time}")
            
            # Show processing summary
            if project_mode == "Batch Projects":
                total_projects = len(valid_projects)
                total_prompts = sum(len(project['prompts']) for project in valid_projects)
                successful_images = len(st.session_state.all_generated_images) if st.session_state.all_generated_images else 0
                
                st.info(f"📊 Batch Processing Summary:")
                st.info(f"   • Total Projects: {total_projects}")
                st.info(f"   • Total Prompts: {total_prompts}")
                st.info(f"   • Images Generated: {successful_images}")
                
                # Show timing efficiency
                if hasattr(st.session_state, 'total_generation_time') and successful_images > 0:
                    time_per_image = st.session_state.total_generation_time / successful_images
                    if time_per_image < 60:
                        efficiency = f"{time_per_image:.1f}s per image"
                    else:
                        minutes = int(time_per_image // 60)
                        seconds = int(time_per_image % 60)
                        efficiency = f"{minutes}m {seconds}s per image"
                    st.info(f"   • **Efficiency:** {efficiency}")
                
                # Show project breakdown
                st.markdown("**📋 Project Breakdown:**")
                for project in valid_projects:
                    project_title = project['title']
                    project_prompts = project['prompts']
                    project_images = [img for img in st.session_state.all_generated_images if hasattr(img, 'project_title') and img.project_title == project_title]
                    st.info(f"   • **{project_title}**: {len(project_prompts)} prompts → {len(project_images)} images")
                
            else:
                # Single project summary
                total_lines = len(raw_prompts) if 'raw_prompts' in locals() else 0
                valid_lines = len(prompts) if 'prompts' in locals() else 0
                skipped_lines = total_lines - valid_lines
                
                st.info(f"📊 Processing Summary:")
                st.info(f"   • Total lines in input: {total_lines}")
                st.info(f"   • Valid prompts processed: {valid_lines}")
                st.info(f"   • Empty lines skipped: {skipped_lines}")
                
                # Show timing efficiency for single project
                if hasattr(st.session_state, 'total_generation_time') and st.session_state.all_generated_images:
                    successful_images = len(st.session_state.all_generated_images)
                    if successful_images > 0:
                        time_per_image = st.session_state.total_generation_time / successful_images
                        if time_per_image < 60:
                            efficiency = f"{time_per_image:.1f}s per image"
                        else:
                            minutes = int(time_per_image // 60)
                            seconds = int(time_per_image % 60)
                            efficiency = f"{minutes}m {seconds}s per image"
                        st.info(f"   • **Efficiency:** {efficiency}")
            
            if st.session_state.all_generated_images:
                st.success(f"✅ Successfully generated {len(st.session_state.all_generated_images)} total images!")
            else:
                st.error("❌ No images were generated successfully!")
            
            # Show failed prompts summary
            if st.session_state.failed_prompts:
                st.markdown("---")
                st.markdown("### ⚠️ Failed Prompts Summary")
                
                st.markdown(f"""
                <div class="failed-prompt-box">
                <strong>❌ {len(st.session_state.failed_prompts)} prompt(s) failed:</strong><br>
                """, unsafe_allow_html=True)
                
                for failed in st.session_state.failed_prompts:
                    # Truncate long prompts for display
                    display_prompt = failed['prompt'][:100] + "..." if len(failed['prompt']) > 100 else failed['prompt']
                    display_error = failed['error'][:200] + "..." if len(failed['error']) > 200 else failed['error']
                    project_info = f" (Project: {failed.get('project_title', 'Unknown')})" if 'project_title' in failed else ""
                    
                    st.markdown(f"""
                    <strong>Prompt {failed['index']}{project_info}:</strong> {display_prompt}<br>
                    <strong>Error:</strong> {display_error}<br><br>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show failed prompts in a text area for easy copying (only prompts, no errors)
                failed_prompts_only = [f['prompt'] for f in st.session_state.failed_prompts]
                failed_text = "\n".join(failed_prompts_only)
                st.text_area(
                    "📋 Failed Prompts (Copy if needed):",
                    value=failed_text,
                    height=150,
                    help="Copy these failed prompts to retry later (only prompts, no errors)"
                )
            else:
                st.success("🎉 All prompts completed successfully!")
            
            # Add option to retry failed prompts
            if st.session_state.failed_prompts:
                st.markdown("---")
                st.markdown("### 🔄 Retry Failed Prompts")
                if st.button("🔄 Retry Failed Prompts", width='stretch'):
                    # Clear failed prompts and continue with remaining prompts
                    st.session_state.failed_prompts = []
                    st.rerun()
    
    with col2:
        st.header("📊 Status")
        
        if cookie or auth_token or cookie_file:
            st.success("✅ Credentials provided")
        else:
            st.warning("⚠️ No credentials provided")
        
        if project_mode == "Batch Projects":
            if hasattr(st.session_state, 'project_titles') and st.session_state.project_titles:
                st.info(f"📚 Batch Projects: {len(st.session_state.project_titles)} projects")
                for i, project_title in enumerate(st.session_state.project_titles):
                    if hasattr(st.session_state, 'project_prompts') and i < len(st.session_state.project_prompts):
                        prompt_count = len([p.strip() for p in st.session_state.project_prompts[i].split('\n') if p.strip()])
                        st.info(f"   • Project {i+1}: {project_title} ({prompt_count} prompts)")
        else:
            if prompts_text:
                raw_lines = prompts_text.split('\n')
                valid_prompts = [p.strip() for p in raw_lines if p.strip()]
                st.info(f"📝 Prompts: {len(valid_prompts)} valid prompt(s) out of {len(raw_lines)} lines")
            
            if title:
                st.info(f"🎯 Title: {title}")
        
        # Show failed prompts count
        if st.session_state.failed_prompts:
            st.warning(f"⚠️ Failed: {len(st.session_state.failed_prompts)} prompt(s)")
        
        # Show generation status
        if (st.session_state.generation_complete and 
            st.session_state.show_download_button and
            st.session_state.all_generated_images and
            len(st.session_state.all_generated_images) > 0 and
            not st.session_state.generating and
            hasattr(st.session_state, 'current_title') and 
            st.session_state.current_title and
            not st.session_state.get('clearing_state', False)):
            image_count = len(st.session_state.all_generated_images)
            st.success(f"✅ Generation Complete: {image_count} images")
            
            # Show timing information
            if hasattr(st.session_state, 'total_generation_time'):
                total_time = st.session_state.total_generation_time
                if total_time < 60:
                    time_display = f"{total_time:.1f}s"
                elif total_time < 3600:
                    minutes = int(total_time // 60)
                    seconds = int(total_time % 60)
                    time_display = f"{minutes}m {seconds}s"
                else:
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    time_display = f"{hours}h {minutes}m"
                
                st.info(f"⏱️ **Time:** {time_display}")
            
            # Show download all reminder
            st.info("💡 Use the 'Download All Images (ZIP)' button above to download all images at once!")
        
        # Environment info
        st.markdown("---")
        st.markdown("**Environment Info:**")
        st.code(f"Model: {actual_model}\nAspect: {aspect_ratio}\nCount: {count}\nSeed: {seed or 'Random'}")
        
        # Quick actions
        st.markdown("---")
        st.markdown("**Quick Actions:**")
        
        # Download All Images Button in sidebar (only show after new generation)
        if (st.session_state.generation_complete and 
            st.session_state.show_download_button and  # New flag to control when to show download
            st.session_state.all_generated_images and 
            len(st.session_state.all_generated_images) > 0 and  # Ensure there are actually images
            not st.session_state.generating and  # Don't show while generating
            hasattr(st.session_state, 'current_title') and 
            st.session_state.current_title and  # Only show if we have a valid title
            not st.session_state.get('clearing_state', False)):  # Don't show while clearing
            
            st.markdown("### 📥 Download All Images")
            
            # Create ZIP with project organization
            if project_mode == "Batch Projects" and hasattr(st.session_state, 'project_titles'):
                # For batch projects, organize by project
                zip_data = create_batch_zip_file(st.session_state.all_generated_images, st.session_state.project_titles)
                zip_filename = "batch_projects_all_images.zip"
            else:
                # For single project
                zip_data = create_zip_file(st.session_state.all_generated_images, title if title else "images")
                safe_title = "".join(c for c in (title if title else "images") if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')
                zip_filename = f"{safe_title}_all_images.zip"
            
            if zip_data:
                st.download_button(
                    label="📦 Download All Images (ZIP)",
                    data=zip_data,
                    file_name=zip_filename,
                    mime="application/zip",
                    width='stretch',
                    key="download_all_sidebar"
                )
                st.info(f"📁 Contains {len(st.session_state.all_generated_images)} images")
        
        if st.button("🔄 Clear All", width='stretch'):
            try:
                # Set clearing state flag to prevent any downloads
                st.session_state.clearing_state = True
                
                # Completely reset all generation-related state
                reset_all_session_state()
                
                # Force immediate state update before rerun
                st.session_state.clear()
                
                # Reinitialize with clean defaults
                st.session_state.failed_prompts = []
                st.session_state.generation_complete = False
                st.session_state.all_generated_images = []
                st.session_state.current_title = ""
                st.session_state.show_download_button = False
                st.session_state.generating = False
                st.session_state.clearing_state = False  # Reset clearing flag
                
                st.rerun()
            except Exception as clear_error:
                st.error(f"❌ Failed to clear state: {clear_error}")
                # Reset clearing flag if error occurs
                st.session_state.clearing_state = False
        
        if st.button("💾 Save Settings", width='stretch'):
            # Save current settings to session state
            st.session_state.saved_settings = {
                "title": title,
                "prompts": prompts_text,
                "model": model,
                "aspect_ratio": aspect_ratio,
                "count": count,
                "seed": seed
            }
            st.success("Settings saved!")

if __name__ == "__main__":
    main()
