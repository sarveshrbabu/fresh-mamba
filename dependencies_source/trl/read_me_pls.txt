The dependencies for this implementation to work are quite odd. 

You need exactly pytorch 2.2.0 with cuda 12 backend for Xformers to download. 

You need this monkeypatched version of the MAMBA library from source. 

You need to install from source: trl and transformers. 
