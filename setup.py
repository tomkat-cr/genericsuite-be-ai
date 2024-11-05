from setuptools import setup

version = '0.1.12'
description = 'The GenericSuite AI for Python (backend version).'
long_description = '''
The GenericSuite AI
===================

GenericSuite AI (backend version) is a versatile backend solution, designed to
provide a comprehensive suite of features, tools and functionalities for
AI oriented Python APIs.
'''.lstrip()

# https://pypi.org/classifiers/

classifiers = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: ISC License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    "Operating System :: OS Independent",
    'Topic :: Software Development',
]

setup(
    name='genericsuite_ai',
    python_requires='>=3.9,<4.0',
    version=version,
    description=description,
    long_description=long_description,
    author='Carlos J. Ramirez',
    author_email='tomkat_cr@yahoo.com',
    url='https://github.com/tomkat-cr/genericsuite-be-ai',
    license='ISC License',
    py_modules=['genericsuite_ai'],
    classifiers=classifiers,
)
