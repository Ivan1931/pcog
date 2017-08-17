import sys
from json import loads
from .agent import simulate
from .envconf import Action
import logging
import SocketServer
from threading import BoundedSemaphore
import datetime

logging.basicConfig(filename="pcog.log", filemode="w", level=logging.INFO)
logging.info("Session begins: {}".format(datetime.datetime.now()))

class PCogHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        logging.info("Handling pcog connection request")
        self.data = self.rfile.readline().strip()
        while self.data:
            logging.info("{} wrote:".format(self.client_address[0]))
            logging.info(self.data)
            action = simulate(self.data)
            logging.info("Selected action: {} = {}".format(Action.qcog_action_name(action), action))
            self.wfile.write("{}\n".format(action))
            self.wfile.flush()
            self.data = self.rfile.readline().strip()


def main(args=None):
    HOST, PORT = "localhost", 9999
    server = SocketServer.TCPServer((HOST, PORT), PCogHandler)
    server.serve_forever()


if __name__=="__main__":
    main(sys.argv[1:])
