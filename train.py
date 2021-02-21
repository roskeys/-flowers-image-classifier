import argparse
import logging
import time
from utils_ic import load_data, read_jason
from model_ic import NN_Classifier, validation, make_NN, save_checkpoint

logger = logging.getLogger()

logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)15s %(levelname)5s: %(message)s')

stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
stream.setFormatter(formatter)
logger.addHandler(stream)

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")
parser.add_argument("--scratch", default=False, help="Start from scratch")
parser.add_argument("--all", default=False, help="Train all parameters of the model")

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

handler = logging.FileHandler(
    f'logs/{args.arch}{"_all" if args.all else ""}{"_scr" if args.scratch else ""}-{time.strftime("%d-%H-%M-%S", time.localtime(time.time()))}.log')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

model = make_NN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate,
                device=args.gpu, model_name=args.arch, trainloader=trainloader, validloader=validloader,
                testloader=testloader, train_data=train_data, from_scratch=args.scratch, train_all_parameters=args.all,
                logger=logger)

if args.save_dir:
    save_checkpoint(model, args.save_dir)
