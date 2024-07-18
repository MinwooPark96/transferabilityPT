from setuptools import setup, find_packages

setup(
    name='transferability',
    version='0.1.0',
    description='pre-trained soft prompt can across pre-trained models!',
    author='Min Woo Park',
    author_email='alsdn0110@snu.ac.kr',
    packages=find_packages(where='src'),  # src 폴더 내부의 패키지들을 찾습니다
    package_dir={'': 'src'}, 
    # install_requires=[
    #     'numpy',
    #     'pandas',
    #     'torch',
    #     'transformers',
    #     # 여기에 필요한 다른 라이브러리들을 추가합니다
    # ],
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
    python_requires='>=3.11',
)