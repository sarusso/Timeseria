# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Timeseria'
copyright = 'Stefano Alberto Russo'
author = 'Stefano Alberto Russo'

# The full version, including alpha/beta/rc tags
release = 'v2.3.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary']
autodoc_default_flags = ['members']
autosummary_generate = False
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__call__'
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

def custom_skip_member(app, what, name, obj, skip, options):
    #print('app="{}", what="{}", name="{}", obj="{}", skip="{}", options={}\n\n'.format(app, what, name, obj, skip, options))
    
    # Exclude all private members including the __call__ which would otherwise be included
    if name.startswith('_'):
        return True
    
    # Exclude exceptions unnecessary methods
    if name in ['args', 'with_traceback']:
        return True
    
    return None


def setup(app):
    app.connect('autodoc-skip-member', custom_skip_member)
    app.add_css_file('css/custom.css')
    
