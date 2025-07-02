from setuptools import setup
import os
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Event Horizon CLI shell"

setup(
    name="EH_shell",
    version="1_0_4a",
    author="QUIK",
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
