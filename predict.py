from eyeball.predictor import Predictor
from argparse import ArgumentParser
from PIL import Image
import icecream
icecream.install()

parser = ArgumentParser()
parser.add_argument("-c", dest="config", required=True)
parser.add_argument("-i", dest="image", required=True)
args = parser.parse_args()
image = Image.open(args.image)

predictor = Predictor.from_config(args.config)
result = predictor.predict_single(image)
ic(result)
