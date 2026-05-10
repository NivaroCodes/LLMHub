from setuptools import setup, find_packages

setup(
    name="llmhub",
    version="0.1.0",
    description="LLMHub Python SDK - Multi-provider LLM gateway",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="LLMHub Contributors",
    author_email="admin@yourdomain.com",
    url="https://github.com/NivaroCodes/LLMHub",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
