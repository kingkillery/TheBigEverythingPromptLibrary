#!/usr/bin/env python3
"""
Repository maintenance script for The Big Everything Prompt Library.

Functions:
- Validate all markdown links in README files
- Check for missing README files in directories
- Generate file counts and statistics
- Update MasterREADME.md with current statistics
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def get_repo_root() -> Path:
    """Find the repository root directory."""
    current = Path(__file__).parent.parent
    if (current / "README.md").exists():
        return current
    raise Exception("Could not find repository root")

def scan_directory_structure(repo_root: Path) -> Dict[str, Dict]:
    """Scan repository structure and count files by type."""
    structure = defaultdict(lambda: {"md_files": 0, "other_files": 0, "subdirs": [], "has_readme": False})
    
    for root, dirs, files in os.walk(repo_root):
        rel_path = Path(root).relative_to(repo_root)
        key = str(rel_path) if rel_path != Path(".") else "root"
        
        # Count files
        md_count = len([f for f in files if f.endswith('.md')])
        other_count = len([f for f in files if not f.endswith('.md')])
        
        structure[key]["md_files"] = md_count
        structure[key]["other_files"] = other_count
        structure[key]["subdirs"] = dirs
        structure[key]["has_readme"] = "README.md" in files
        
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
    
    return dict(structure)

def find_missing_readmes(structure: Dict[str, Dict]) -> List[str]:
    """Find directories that should have README files but don't."""
    missing = []
    major_dirs = ["Articles", "CustomInstructions", "Guides", "Jailbreak", "Security", "SystemPrompts"]
    
    for path, info in structure.items():
        if path == "root":
            continue
        
        # Check if it's a major directory or has significant content
        is_major = any(path.startswith(major) for major in major_dirs)
        has_content = info["md_files"] > 3 or len(info["subdirs"]) > 0
        
        if (is_major or has_content) and not info["has_readme"]:
            missing.append(path)
    
    return missing

def validate_markdown_links(file_path: Path, repo_root: Path) -> List[str]:
    """Check if markdown links in a file are valid."""
    broken_links = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return [f"Could not read file: {e}"]
    
    # Find markdown links [text](path)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(link_pattern, content)
    
    for text, link in matches:
        # Skip external URLs and anchors
        if link.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        
        # Handle relative paths
        if link.startswith('./'):
            link = link[2:]
        
        # Decode URL encoding (handle %20 spaces)
        import urllib.parse
        link = urllib.parse.unquote(link)
        
        # Resolve path relative to file location
        if link.startswith('/'):
            target_path = repo_root / link[1:]
        else:
            target_path = file_path.parent / link
        
        # Check if target exists
        if not target_path.exists():
            broken_links.append(f"Broken link in {file_path.name}: [{text}]({link})")
    
    return broken_links

def generate_statistics_report(structure: Dict[str, Dict]) -> Dict[str, int]:
    """Generate summary statistics for the repository."""
    stats = {
        "total_md_files": 0,
        "total_other_files": 0,
        "custom_instructions": 0,
        "system_prompts": 0,
        "guides": 0,
        "articles": 0,
        "security_files": 0,
        "jailbreak_files": 0,
        "missing_readmes": 0
    }
    
    for path, info in structure.items():
        stats["total_md_files"] += info["md_files"]
        stats["total_other_files"] += info["other_files"]
        
        if path.startswith("CustomInstructions"):
            stats["custom_instructions"] += info["md_files"]
        elif path.startswith("SystemPrompts"):
            stats["system_prompts"] += info["md_files"]
        elif path.startswith("Guides"):
            stats["guides"] += info["md_files"]
        elif path.startswith("Articles"):
            stats["articles"] += info["md_files"]
        elif path.startswith("Security"):
            stats["security_files"] += info["md_files"]
        elif path.startswith("Jailbreak"):
            stats["jailbreak_files"] += info["md_files"]
        
        if not info["has_readme"] and info["md_files"] > 0:
            stats["missing_readmes"] += 1
    
    return stats

def check_repo_health(repo_root: Path) -> Dict:
    """Perform comprehensive repository health check."""
    print("🔍 Scanning repository structure...")
    structure = scan_directory_structure(repo_root)
    
    print("📊 Generating statistics...")
    stats = generate_statistics_report(structure)
    
    print("🔗 Checking for missing README files...")
    missing_readmes = find_missing_readmes(structure)
    
    print("🔍 Validating key markdown links...")
    broken_links = []
    key_files = [
        repo_root / "README.md",
        repo_root / "MasterREADME.md",
        repo_root / "CustomInstructions" / "README.md",
        repo_root / "SystemPrompts" / "README.md",
        repo_root / "Guides" / "README.md"
    ]
    
    for file_path in key_files:
        if file_path.exists():
            broken_links.extend(validate_markdown_links(file_path, repo_root))
    
    return {
        "structure": structure,
        "stats": stats,
        "missing_readmes": missing_readmes,
        "broken_links": broken_links
    }

def print_health_report(health_data: Dict):
    """Print a formatted health report."""
    stats = health_data["stats"]
    missing_readmes = health_data["missing_readmes"]
    broken_links = health_data["broken_links"]
    
    print("\n" + "="*60)
    print("📋 REPOSITORY HEALTH REPORT")
    print("="*60)
    
    print(f"\n📊 Statistics:")
    print(f"  Total Markdown Files: {stats['total_md_files']:,}")
    print(f"  Total Other Files: {stats['total_other_files']:,}")
    print(f"  Custom Instructions: {stats['custom_instructions']:,}")
    print(f"  System Prompts: {stats['system_prompts']:,}")
    print(f"  Guides: {stats['guides']:,}")
    print(f"  Articles: {stats['articles']:,}")
    print(f"  Security Files: {stats['security_files']:,}")
    print(f"  Jailbreak Files: {stats['jailbreak_files']:,}")
    
    print(f"\n⚠️  Issues Found:")
    print(f"  Missing README files: {len(missing_readmes)}")
    print(f"  Broken links: {len(broken_links)}")
    
    if missing_readmes:
        print(f"\n📁 Directories missing README files:")
        for path in missing_readmes:
            print(f"  - {path}")
    
    if broken_links:
        print(f"\n🔗 Broken links found:")
        for link in broken_links[:10]:  # Show first 10
            print(f"  - {link}")
        if len(broken_links) > 10:
            print(f"  ... and {len(broken_links) - 10} more")
    
    if not missing_readmes and not broken_links:
        print(f"\n✅ Repository health looks good!")

def main():
    """Main function."""
    try:
        repo_root = get_repo_root()
        print(f"🏠 Repository root: {repo_root}")
        
        health_data = check_repo_health(repo_root)
        print_health_report(health_data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())