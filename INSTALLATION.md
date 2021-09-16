We have include all dependent libraries into [setup.cfg](./setup.cfg#L18), so that they will be automatically installed when you install `cl-lite` through `pip`. You are free to manually install your prefered version of these dependencies unless they conflict with `cl-lite`.

Though this library has not been published to PyPI, you can install it from GitHub or a local clone of `cl-lite`. 

## Install from GitHub 

This is the easiest and the recommend way to use `cl-lite` without modification, just run the following command in any shell:

```bash
pip install git+https://github.com/gqk/cl-lite.git
``` 

or run the following command to install a specific version:

```bash
pip install git+https://github.com/gqk/cl-lite.git@v0.5.0
```

Though installed as a global library, you are allowed to flexiblely derive or override any class to fit you requirements. You can find some useful exemples in [cl-lite-projects](https://github.com/gqk/cl-lite-projects).

Please refer to pip's [VCS Support](https://pip.pypa.io/en/stable/topics/vcs-support) for more information. 


## Install from a local clone

If you want directly modify `cl-lite` for testing or other purposes, you can install it from a local clone with `-e`:

```bash
git clone https://github.com/gqk/cl-lite.git /path/to/cl-lite

pip install -e /path/to/cl-lite
``` 

Please refer to pip's [document](https://pip.pypa.io/en/stable/cli/pip_install) for more information. 



