#!/usr/bin/env python3
"""
ImageFX CLI - Command Line Interface
Unofficial reverse engineered API for imageFX service provided by Google Labs

Usage:
    python imagefx_cli.py --prompt "purple cat" --cookie "$COOKIE"
    python imagefx_cli.py --prompt "purple cat" --auth "$TOKEN"
    python imagefx_cli.py --prompt "purple cat" --cookie-file "cookies.txt"
    python imagefx_cli.py --title "My Project" --prompts-file "prompts.txt" --cookie-file "cookies.txt"
"""

import argparse
import sys
import os
import random
from pathlib import Path
from imagefx import ImageFX, Credentials, Prompt, save_images

def read_prompts_from_file(file_path: str) -> list:
    """Read prompts from a text file, one per line"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    except Exception as e:
        print(f"Error reading prompts file: {e}", file=sys.stderr)
        return []

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
            ("IMAGE_ASPECT_RATIO_LANDSCAPE", "Landscape (4:3)"),
            ("IMAGE_ASPECT_RATIO_SQUARE", "Square (1:1)"),
            ("IMAGE_ASPECT_RATIO_PORTRAIT", "Portrait (3:4)"),
            ("IMAGE_ASPECT_RATIO_UNSPECIFIED", "Unspecified (Let model decide)"),
        ]
        return aspect_ratios, "IMAGEN_2", "IMAGE_ASPECT_RATIO_LANDSCAPE"
    
    else:
        # IMAGEN_4, IMAGEN_3_1, IMAGEN_3_5 use standard aspect ratios
        aspect_ratios = [
            ("IMAGE_ASPECT_RATIO_LANDSCAPE", "Landscape (4:3)"),
            ("IMAGE_ASPECT_RATIO_SQUARE", "Square (1:1)"),
            ("IMAGE_ASPECT_RATIO_PORTRAIT", "Portrait (3:4)"),
            ("IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE", "Landscape (4:3) - Explicit"),
            ("IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR", "Portrait (3:4) - Explicit"),
            ("IMAGE_ASPECT_RATIO_UNSPECIFIED", "Unspecified (Let model decide)"),
        ]
        return aspect_ratios, model, "IMAGE_ASPECT_RATIO_LANDSCAPE"

def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Google's ImageFX service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "purple cat" --cookie "$COOKIE"
  %(prog)s --prompt "purple cat" --auth "$TOKEN"
  %(prog)s --prompt "purple cat" --cookie-file "cookies.txt"
  %(prog)s --title "My Project" --prompts-file "prompts.txt" --cookie-file "cookies.txt"
  
IMAGEN_3 Aspect Ratio Models:
  --model IMAGEN_3 --ratio IMAGEN_3_LANDSCAPE     # Landscape (4:3)
  --model IMAGEN_3 --ratio IMAGEN_3_PORTRAIT      # Portrait (3:4)
  --model IMAGEN_3 --ratio IMAGEN_3_PORTRAIT_THREE_FOUR    # Portrait (3:4)
  --model IMAGEN_3 --ratio IMAGEN_3_LANDSCAPE_FOUR_THREE  # Landscape (4:3)
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="ImageFX CLI 1.1.1"
    )
    
    parser.add_argument(
        "--title",
        help="Project title for file naming (e.g., 'My Project')"
    )
    
    parser.add_argument(
        "--prompt",
        help="Single prompt for image generation (use --prompts-file for multiple prompts)"
    )
    
    parser.add_argument(
        "--prompts-file",
        help="Text file with prompts, one per line"
    )
    
    parser.add_argument(
        "--auth",
        help="Authentication token for generating images"
    )
    
    parser.add_argument(
        "--cookie",
        help="Cookie (for auto auth token generation)"
    )
    
    parser.add_argument(
        "--cookie-file",
        help="Path to Netscape cookie file (e.g., cookies.txt)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reference image (default: random)"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        choices=range(1, 11),
        help="Number of images to generate (default: 4)"
    )
    
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory to save images (default: current directory)"
    )
    
    parser.add_argument(
        "--model",
        default="IMAGEN_4",
        choices=[
            "IMAGEN_2", "IMAGEN_3", "IMAGEN_4", "IMAGEN_3_1", "IMAGEN_3_5"
        ],
        help="Model to use for generation (default: IMAGEN_4)"
    )
    
    parser.add_argument(
        "--ratio",
        default="IMAGE_ASPECT_RATIO_LANDSCAPE",
        help="Aspect ratio or model variant (see examples for IMAGEN_3)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.auth and not args.cookie and not args.cookie_file:
        print("Error: Either --auth, --cookie, or --cookie-file must be provided", file=sys.stderr)
        sys.exit(1)
    
    if not args.prompt and not args.prompts_file:
        print("Error: Either --prompt or --prompts-file must be provided", file=sys.stderr)
        sys.exit(1)
    
    if args.cookie and len(args.cookie) < 70:
        print("Error: Invalid cookie provided (too short)", file=sys.stderr)
        sys.exit(1)
    
    if args.cookie_file and not Path(args.cookie_file).exists():
        print(f"Error: Cookie file not found: {args.cookie_file}", file=sys.stderr)
        sys.exit(1)
    
    if args.prompts_file and not Path(args.prompts_file).exists():
        print(f"Error: Prompts file not found: {args.prompts_file}", file=sys.stderr)
        sys.exit(1)
    
    # Get prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        prompts = read_prompts_from_file(args.prompts_file)
        if not prompts:
            print("Error: No valid prompts found in prompts file", file=sys.stderr)
            sys.exit(1)
    
    # Set default title if not provided
    title = args.title or "Generated_Images"
    
    # Validate aspect ratio for IMAGEN_3
    if args.model == "IMAGEN_3":
        valid_imagen3_ratios = [
            "IMAGEN_3_LANDSCAPE", "IMAGEN_3_PORTRAIT", 
            "IMAGEN_3_PORTRAIT_THREE_FOUR", "IMAGEN_3_LANDSCAPE_FOUR_THREE"
        ]
        if args.ratio not in valid_imagen3_ratios:
            print(f"Error: For IMAGEN_3, --ratio must be one of: {', '.join(valid_imagen3_ratios)}", file=sys.stderr)
            print("IMAGEN_3 uses separate models for different aspect ratios.", file=sys.stderr)
            sys.exit(1)
    
    # Create credentials
    credentials = Credentials(
        cookie=args.cookie,
        authorization_key=args.auth,
        cookie_file=args.cookie_file
    )
    
    try:
        # Initialize ImageFX
        print("Initializing ImageFX...")
        imagefx = ImageFX(credentials)
        
        all_generated_images = []
        
        # Process each prompt
        for prompt_index, prompt_text in enumerate(prompts):
            print(f"\n🎨 Processing prompt {prompt_index + 1}/{len(prompts)}: '{prompt_text}'")
            
            # Create prompt object
            prompt = Prompt(
                prompt=prompt_text.strip(),
                count=args.count,
                seed=args.seed,
                model=args.model,
                aspect_ratio=args.ratio
            )
            
            # Generate images
            print(f"Generating {args.count} image(s) with prompt: '{prompt_text}'")
            print(f"Model: {args.model}, Aspect Ratio/Model Variant: {args.ratio}")
            if args.seed is not None:
                print(f"Seed: {args.seed}")
            
            result = imagefx.generate_image(prompt)
            
            if "Err" in result:
                print(f"Error generating images for prompt '{prompt_text}': {result['Err']}", file=sys.stderr)
                continue
            
            # Add prompt info to images
            for img in result['Ok']:
                img.prompt = prompt_text  # Override with the actual prompt used
            
            all_generated_images.extend(result['Ok'])
            print(f"✅ Successfully generated {len(result['Ok'])} images for prompt {prompt_index + 1}")
        
        if not all_generated_images:
            print("Error: No images were generated successfully", file=sys.stderr)
            sys.exit(1)
        
        # Save images
        print(f"\n🎉 Successfully generated {len(all_generated_images)} total images!")
        
        # Ensure directory exists
        save_dir = Path(args.dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate random 3 digits for this batch
        random_digits = ''.join([str(random.randint(0, 9)) for _ in range(3)])
        print(f"📁 Using random identifier: {random_digits}")
        
        # Save images with title-based naming + random 3 digits
        saved_files = []
        for i, image in enumerate(all_generated_images):
            # Create safe filename based on title + random 3 digits
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            filename = f"{safe_title}_{random_digits}_{i + 1}.png"
            
            # Save individual image
            if save_image(image.encoded_image, filename, str(save_dir)):
                saved_files.append(filename)
        
        print(f"Images saved to directory: {save_dir.absolute()}")
        print(f"Naming convention: {safe_title}_{random_digits}_1.png, {safe_title}_{random_digits}_2.png, etc.")
        for filename in saved_files:
            print(f"  - {filename}")
        
        # Show image details
        print("\nImage Details:")
        for i, image in enumerate(all_generated_images):
            print(f"  Image {i+1}:")
            print(f"    Prompt: {image.prompt}")
            print(f"    Seed: {image.seed}")
            print(f"    Model: {image.model_name_type}")
            print(f"    Media Generation ID: {image.media_generation_id}")
            print(f"    Workflow ID: {image.workflow_id}")
            print()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def save_image(image_data: str, filename: str, directory: str = ".") -> bool:
    """Save base64 image data to file"""
    try:
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Decode base64 image
        import base64
        image_bytes = base64.b64decode(image_data)
        
        # Save to file
        filepath = Path(directory) / filename
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        return True
    except Exception as e:
        print(f"Failed to save image {filename}: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    main()
