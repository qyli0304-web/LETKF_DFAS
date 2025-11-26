"""
Lightweight configuration parser for Polyphemus-style CFG files with repeated
sections. Supports parsing, resolving markup references, CRUD operations on
section instances, and saving back to file.
"""
from collections import defaultdict, OrderedDict
import re

class ConfigParser:
    def __init__(self):
        self.config = defaultdict(list)
        self.section_counter = defaultdict(int)

    def parse(self, filename):
        current_section = None
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('%'):
                    continue
                
                section_match = re.match(r'^\[(.+)\]$', line)
                if section_match:
                    section_name = section_match.group(1)
                    self.section_counter[section_name] += 1
                    section_key = f"{section_name}#{self.section_counter[section_name]}"
                    current_section = section_key
                    self.config[current_section] = OrderedDict()
                    continue
                
                if current_section is not None:
                    line = re.split(r'(?<!\\)[#%]', line)[0].strip()
                    if not line:
                        continue
                    
                    pairs = re.findall(r'(\S+)\s*[:=]\s*([^=:]+?)(?=\s+\S+\s*[:=]|$)', line)
                    if pairs:
                        for key, value in pairs:
                            value = value.strip()
                            self.config[current_section][key] = value

    def get(self, section, index=1):
        """Return the dict for the specified section instance (1-indexed)."""
        section_key = f"{section}#{index}"
        return self.config.get(section_key, None)
    
    def set(self, section, key, value, index=1):
        """Set a key-value pair in the specified section instance."""
        section_key = f"{section}#{index}"
        if section_key in self.config:
            self.config[section_key][key] = value
        else:
            new_section = OrderedDict()
            new_section[key] = value
            self.config[section_key] = new_section
    
    def delete(self, section, key, index=1):
        """Delete a key from the specified section instance, if present."""
        section_key = f"{section}#{index}"
        if section_key in self.config and key in self.config[section_key]:
            del self.config[section_key][key]

    def copy_section(self, section, index=1):
        """Duplicate a section instance and append it after the last same-type section."""
        section_key = f"{section}#{index}"
        if section_key in self.config:
            self.section_counter[section] += 1
            new_section_key = f"{section}#{self.section_counter[section]}"
            new_config = OrderedDict()
            
            last_section_index = None
            keys = list(self.config.keys())
            for i, key in enumerate(keys):
                if key.startswith(f"{section}#"):
                    last_section_index = i
            
            self.config[new_section_key] = self.config[section_key].copy()
            
            new_config.update((k, self.config[k]) for k in keys[:last_section_index + 1])
            new_config[new_section_key] = self.config[new_section_key]
            new_config.update((k, self.config[k]) for k in keys[last_section_index + 1:])
            self.config = new_config

    def add_section(self, section_name):
        """Add a new section with an automatically incremented index."""
        self.section_counter[section_name] += 1
        section_key = f"{section_name}#{self.section_counter[section_name]}"
        self.config[section_key] = OrderedDict()
        return section_key  

    def add_key(self, section, key, value, index=1):
        """Add or update a key-value pair within a section instance."""
        section_key = f"{section}#{index}"
        if section_key not in self.config:
            self.add_section(section)
            section_key = f"{section}#{self.section_counter[section]}"
        
        self.config[section_key][key] = value

    def save(self, filename):
        """Write the configuration back to a file in standard INI-like format."""
        with open(filename, 'w', encoding='utf-8') as f:
            for section, instance in self.config.items():
                base_section = section.rsplit("#", 1)[0]  
                f.write(f'[{base_section}]\n')
                for key, value in instance.items():
                    f.write(f'{key} = {value}\n')
                f.write('\n')  

    def resolve_markups(self):
        """Resolve placeholders of the form <markup> using values from any section."""
        for section in self.config:
            for key, value in self.config[section].items():
                if '<' in value and '>' in value:
                    for markup in re.findall(r'<([^>]+)>', value):
                        for sec in self.config:
                            if markup in self.config[sec]:
                                value = value.replace(f'<{markup}>', self.config[sec][markup])
                                break
                    self.config[section][key] = value

    def __repr__(self):
        return str(self.config)
    
    def get_section(self):
        """Return a list of all section dicts in insertion order."""
        sections = []
        for section in self.config:
            d = self.config[section]
            sections.append(d)
        return sections