"""
ImageFX API - Python Module
Unofficial reverse engineered API for imageFX service provided by Google Labs

This module provides the same functionality as the TypeScript version but in Python.
"""

import requests
import base64
import json
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import os
from pathlib import Path
from cookie_parser import CookieParser

# Data classes for type safety
@dataclass
class Credentials:
    """Credentials for authentication"""
    cookie: Optional[str] = None
    authorization_key: Optional[str] = None
    cookie_file: Optional[str] = None

@dataclass
class Prompt:
    """Image generation prompt configuration"""
    prompt: str
    count: int = 4
    seed: Optional[int] = None
    model: str = "IMAGEN_4"
    aspect_ratio: str = "IMAGE_ASPECT_RATIO_LANDSCAPE"

@dataclass
class GeneratedImage:
    """Generated image result"""
    encoded_image: str
    seed: int
    media_generation_id: str
    is_mask_edited_image: bool
    prompt: str
    model_name_type: str
    workflow_id: str
    fingerprint_log_record_id: str

class ImageFX:
    """
    Python implementation of the ImageFX API
    
    This class provides the same functionality as the TypeScript ImageFX class:
    - Authentication via cookie or auth token
    - Automatic token generation from cookies
    - Support for Netscape cookie files
    - Image generation with various models and settings
    """
    
    def __init__(self, credentials: Credentials):
        """
        Initialize ImageFX with credentials
        
        Args:
            credentials: Credentials object containing either cookie, authorization_key, or cookie_file
            
        Raises:
            ValueError: If all authentication methods are missing or invalid
        """
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
        
        # Add the missing header if cookie doesn't start with the expected prefix
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
                print(f"✅ Loaded session token from cookie file: {self.credentials.cookie_file}")
            else:
                raise ValueError("No session token found in cookie file")
                
        except Exception as e:
            raise ValueError(f"Failed to load cookies from file: {e}")
    
    def _make_request(self, url: str, method: str = "GET", headers: Optional[Dict] = None, 
                     body: Optional[str] = None) -> Dict[str, Any]:
        """
        Make HTTP request with proper headers
        
        Args:
            url: Request URL
            method: HTTP method (GET or POST)
            headers: Additional headers
            body: Request body for POST requests
            
        Returns:
            Dict with either "Ok" (success) or "Err" (error) key
        """
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
        """
        Check and validate authentication token
        
        Returns:
            Dict with either "Ok" (success) or "Err" (error) key
        """
        if not self.credentials.authorization_key and not self.credentials.cookie:
            return {"Err": "Authorization token and Cookie both are missing."}
        
        if self.credentials.cookie and not self.credentials.authorization_key:
            # Get auth token internally
            result = self.get_auth_token(mutate=True)
            if "Err" in result:
                return result
        
        return {"Ok": True}
    
    def get_auth_token(self, mutate: bool = False) -> Dict[str, Any]:
        """
        Get authentication token from cookie
        
        Args:
            mutate: If True, update the credentials object with the new token
            
        Returns:
            Dict with either "Ok" (token) or "Err" (error) key
        """
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
        """
        Generate image from provided prompt
        
        Args:
            prompt: Prompt object containing generation parameters
            
        Returns:
            Dict with either "Ok" (list of GeneratedImage) or "Err" (error) key
        """
        token_res = self.check_token()
        if "Err" in token_res:
            return token_res
        
        # IMAGEN_3_5 is behind the scene IMAGEN_4
        if prompt.model == "IMAGEN_4":
            prompt.model = "IMAGEN_3_5"
        
        request_body = {
            "userInput": {
                "candidatesCount": prompt.count or 4,
                "prompts": [prompt.prompt],
                "seed": prompt.seed or 0,
            },
            "aspectRatio": prompt.aspect_ratio or "IMAGE_ASPECT_RATIO_LANDSCAPE",
            "modelInput": {"modelNameType": prompt.model or "IMAGEN_3_5"},
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
            parsed_res = json.loads(result["Ok"])
            images = parsed_res["imagePanels"][0]["generatedImages"]
            
            if not isinstance(images, list):
                return {"Err": f"Invalid response received: {result['Ok']}"}
            
            # Convert to GeneratedImage objects
            generated_images = []
            for img in images:
                generated_images.append(GeneratedImage(
                    encoded_image=img["encodedImage"],
                    seed=img["seed"],
                    media_generation_id=img["mediaGenerationId"],
                    is_mask_edited_image=img["isMaskEditedImage"],
                    prompt=img["prompt"],
                    model_name_type=img["modelNameType"],
                    workflow_id=img["workflowId"],
                    fingerprint_log_record_id=img["fingerprintLogRecordId"]
                ))
            
            return {"Ok": generated_images}
            
        except (json.JSONDecodeError, KeyError) as e:
            return {"Err": f"Failed to parse JSON: {result['Ok']}"}

def save_image(image_data: str, filename: str, directory: str = ".") -> bool:
    """
    Save base64 image data to file
    
    Args:
        image_data: Base64 encoded image data
        filename: Name of the file to save
        directory: Directory to save the file in (default: current directory)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Save to file
        filepath = Path(directory) / filename
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        return True
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False

def save_images(images: List[GeneratedImage], directory: str = ".", prefix: str = "image") -> List[str]:
    """
    Save multiple generated images to files
    
    Args:
        images: List of GeneratedImage objects
        directory: Directory to save images in
        prefix: Prefix for image filenames
        
    Returns:
        List of successfully saved filenames
    """
    saved_files = []
    
    for i, image in enumerate(images):
        filename = f"{prefix}-{i + 1}.png"
        if save_image(image.encoded_image, filename, directory):
            saved_files.append(filename)
    
    return saved_files

# Example usage function
def example_usage():
    """Example of how to use the ImageFX API"""
    
    # Check if AUTH environment variable is set
    auth_token = os.getenv("AUTH")
    cookie = os.getenv("COOKIE")
    cookie_file = os.getenv("COOKIE_FILE")
    
    if not auth_token and not cookie and not cookie_file:
        print("Please set AUTH, COOKIE, or COOKIE_FILE environment variable")
        return
    
    try:
        # Initialize ImageFX with credentials
        credentials = Credentials(
            authorization_key=auth_token,
            cookie=cookie,
            cookie_file=cookie_file
        )
        imagefx = ImageFX(credentials)
        
        # Create prompt
        prompt = Prompt(
            prompt="An eagle is flying and a person is standing over it",
            count=5,
            seed=None,  # Random seed
            model="IMAGEN_4",
            aspect_ratio="IMAGE_ASPECT_RATIO_LANDSCAPE"
        )
        
        # Generate images
        print("Generating images...")
        result = imagefx.generate_image(prompt)
        
        if "Err" in result:
            print(f"Generation failed: {result['Err']}")
            return
        
        # Save images
        print(f"Successfully generated {len(result['Ok'])} images!")
        saved_files = save_images(result['Ok'], ".", "generated")
        
        for filename in saved_files:
            print(f"Saved: {filename}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    example_usage()
