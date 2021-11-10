from livereload import Server, shell

if __name__ == '__main__':
    server = Server()
    server.watch('*.rst', shell('make html'), delay=1)
    server.watch('modules/*.rst', shell('make html'), delay=1)
    server.watch('modules/*/*.rst', shell('make html'), delay=1)
    server.watch('*.md', shell('make html'), delay=1)
    server.watch('*.py', shell('make html'), delay=1)
    server.watch('*.ipynb', shell('make html'), delay=.1)
    server.watch('examples/*.ipynb', shell('make html'), delay=.1)
    server.watch('_static/*', shell('make html'), delay=1)
    server.watch('_templates/*', shell('make html'), delay=1)
    server.serve(root='_build/html', host="cn240.wexac.weizmann.ac.il", port=8888)
