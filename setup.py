from setuptools import setup, find_packages

setup(
    name="medical-chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.18.0',
        'vncorenlp>=1.0.3',
        'underthesea>=1.3.3',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'tqdm>=4.62.0',
        'omegaconf>=2.1',
        'PyYAML>=5.1',
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'python-dotenv>=0.19.0'
    ]
)