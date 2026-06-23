#!/usr/bin/env python3

import yaml
import json
import os
import re
from typing import Any, Union

def convert_value_to_string(value: Any) -> str:
    """Convert a Python value to a Hydra-compatible string representation."""
    if isinstance(value, (list, dict)):
        # Convert to JSON and escape any single quotes
        return json.dumps(value).replace("'", "\\'")
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Escape single quotes and wrap in single quotes
        #escaped = value.replace("'", r"\'")
        return f"{value}"
    elif value is None:
        return "null"
    else:
        raise ValueError(f"Unsupported type: {type(value)}")

def _substitute_env_vars(config):
    """Recursively substitute environment variables in config values."""
    
    def substitute_value(value):
        if isinstance(value, str):
            # Pattern to match ${oc.env:VARIABLE_NAME}
            pattern = r'\$\{oc\.env:([^}]+)\}'
            return re.sub(pattern, lambda m: os.environ.get(m.group(1), f"${{{m.group(0)}}}"), value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        return value
    
    return substitute_value(config)

def yaml_to_hydra(yaml_data: Union[str, dict], root_key: str = "") -> str:
    """Convert YAML content to a Hydra override string.
    
    Args:
        yaml_data: Either a YAML string or a parsed YAML dictionary
        root_key: The current key path for recursive calls
        
    Returns:
        A Hydra override string
    """
    # Parse YAML if string input
    if isinstance(yaml_data, str):
        yaml_data = yaml.safe_load(yaml_data)
    
    if not isinstance(yaml_data, dict):
        raise ValueError("YAML data must be a dictionary at the root level")
    
    # Handle parsing of env vars like ${oc.env:SHARED_RESOURCE_DIR}
    yaml_data = _substitute_env_vars(yaml_data)
    
    overrides = []
    
    for key, value in yaml_data.items():
        current_key = f"{root_key}.{key}" if root_key else key
        
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            overrides.append(yaml_to_hydra(value, current_key))
        else:
            # Handle leaf nodes (including lists and primitive types)
            override_value = convert_value_to_string(value)
            overrides.append(f"{current_key}={override_value}")
    
    return " ".join(overrides)

def main():
    """CLI interface for the converter."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: yaml_to_hydra.py <yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    try:
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        result = yaml_to_hydra(yaml_content)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 