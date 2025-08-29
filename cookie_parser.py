#!/usr/bin/env python3
"""
Cookie Parser for ImageFX API
Reads Netscape cookie files and extracts authentication tokens
"""

import re
from typing import Optional, Dict, List
from pathlib import Path

class CookieParser:
    """Parse Netscape cookie files and extract authentication tokens"""
    
    @staticmethod
    def parse_cookie_file(file_path: str) -> Dict[str, str]:
        """
        Parse a Netscape cookie file and return a dictionary of cookies
        
        Args:
            file_path: Path to the cookie file
            
        Returns:
            Dictionary mapping cookie names to values
        """
        cookies = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse cookie line
                    parts = line.split('\t')
                    if len(parts) >= 7:
                        domain = parts[0]
                        path = parts[2]
                        secure = parts[3] == 'TRUE'
                        expiration = parts[4]
                        name = parts[5]
                        value = parts[6]
                        
                        # Only include cookies for labs.google domain
                        if domain == 'labs.google':
                            cookies[name] = value
                            
        except Exception as e:
            print(f"Error parsing cookie file: {e}")
            
        return cookies
    
    @staticmethod
    def extract_session_token(cookies: Dict[str, str]) -> Optional[str]:
        """
        Extract the session token from cookies
        
        Args:
            cookies: Dictionary of cookies
            
        Returns:
            Session token if found, None otherwise
        """
        # Try different possible session token keys
        session_keys = [
            '__Secure-next-auth.session-token',
            'session-token',
            'auth-token'
        ]
        
        for key in session_keys:
            if key in cookies:
                return cookies[key]
        
        return None
    
    @staticmethod
    def extract_csrf_token(cookies: Dict[str, str]) -> Optional[str]:
        """
        Extract the CSRF token from cookies
        
        Args:
            cookies: Dictionary of cookies
            
        Returns:
            CSRF token if found, None otherwise
        """
        csrf_keys = [
            '__Host-next-auth.csrf-token',
            'csrf-token'
        ]
        
        for key in csrf_keys:
            if key in cookies:
                return cookies[key]
        
        return None
    
    @staticmethod
    def get_auth_credentials(cookie_file_path: str) -> Dict[str, Optional[str]]:
        """
        Get authentication credentials from a cookie file
        
        Args:
            cookie_file_path: Path to the cookie file
            
        Returns:
            Dictionary with session_token and csrf_token
        """
        cookies = CookieParser.parse_cookie_file(cookie_file_path)
        
        return {
            'session_token': CookieParser.extract_session_token(cookies),
            'csrf_token': CookieParser.extract_csrf_token(cookies),
            'all_cookies': cookies
        }

def main():
    """Test the cookie parser"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python cookie_parser.py <cookie_file_path>")
        sys.exit(1)
    
    cookie_file = sys.argv[1]
    
    if not Path(cookie_file).exists():
        print(f"Cookie file not found: {cookie_file}")
        sys.exit(1)
    
    print(f"Parsing cookie file: {cookie_file}")
    print("=" * 50)
    
    credentials = CookieParser.get_auth_credentials(cookie_file)
    
    if credentials['session_token']:
        print("✅ Session token found!")
        print(f"Token: {credentials['session_token'][:50]}...")
    else:
        print("❌ No session token found")
    
    if credentials['csrf_token']:
        print("✅ CSRF token found!")
        print(f"Token: {credentials['csrf_token'][:50]}...")
    else:
        print("❌ No CSRF token found")
    
    print(f"\nTotal cookies found: {len(credentials['all_cookies'])}")
    
    if credentials['all_cookies']:
        print("\nAll cookies:")
        for name, value in credentials['all_cookies'].items():
            print(f"  {name}: {value[:50]}{'...' if len(value) > 50 else ''}")

if __name__ == "__main__":
    main()
