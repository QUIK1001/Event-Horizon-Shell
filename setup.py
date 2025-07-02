from setuptools import setup
from pathlib import Path
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="event-horizon-quik",
    version="1.0.5",
    author="Your Name",
    description="Cosmic CLI shell with Python",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    py_modules=["v1_0_4a"],
    install_requires=[
        "colorama",
        "psutil",
        "requests",
        "packaging"
    ],
    entry_points={
        "console_scripts": ["eh-shell=v1_0_4a:menu"]
    }
)
