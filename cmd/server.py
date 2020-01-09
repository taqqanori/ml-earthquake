from http.server import SimpleHTTPRequestHandler, HTTPServer
import fire

def main(recipe='recipe.json', out_dir='out', port=8080):
    print('starting server at port {} with recipe {} and output files in {}'.format(port, recipe, out_dir))
    server = HTTPServer(('', port), lambda *args: _RecipeAndLOutHandler(recipe, out_dir, *args))
    server.serve_forever()

class _RecipeAndLOutHandler(SimpleHTTPRequestHandler):
    def __init__(self, recipe, out_dir, *args):
        self.recipe = recipe
        super().__init__(*args, directory=out_dir)

    def translate_path(self, path):
        if path == '/recipe.json':
            return self.recipe
        return super().translate_path(path)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin','*')
        super().end_headers()

if __name__ == '__main__':
    fire.Fire(main)