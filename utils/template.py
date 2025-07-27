import os
from jinja2 import Environment, FileSystemLoader

def get_template_env():
    base_dir     = os.path.dirname(os.path.abspath(__file__)) 
    template_dir = os.path.join(base_dir, '..', 'templates') 
    return Environment(loader=FileSystemLoader(template_dir), autoescape=False)