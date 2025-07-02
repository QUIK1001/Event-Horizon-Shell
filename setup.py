from setuptools import setup

setup(
    name="Event-Horizon-quik", 
    version="1.0.5",  
    author="Your Name",
    description="Cosmic CLI shell with Python",
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
