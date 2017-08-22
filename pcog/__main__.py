"""
Main entry point for PCog. Can open two types of agents - a model learning agent
or a hand crafted POMPD based agent
"""
import sys
import logging
import SocketServer
import datetime
from .agent import simulate
from .envconf import Action
from .perception import perceive, process, perception_reward
from .usm import UtileSuffixMemory, Instance

logging.basicConfig(filename="pcog.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Session begins: {}".format(datetime.datetime.now()))

class PCogHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        logger.info("Handling pcog connection request")
        self.data = self.rfile.readline().strip()
        while self.data:
            logger.info("{} wrote:".format(self.client_address[0]))
            logger.info(self.data)
            action = simulate(self.data)
            logger.info("Selected action: {} = {}".format(Action.qcog_action_name(action), action))
            self.wfile.write("{}\n".format(action))
            self.wfile.flush()
            self.data = self.rfile.readline().strip()


class PCogModelLearnerHandler(SocketServer.StreamRequestHandler):
    """
    Client that accepts messages from PCog model learning agent.
    It accepts messages where each line is a json string. It watches the change
    and passes those changes onto a usm memory module which is able to plan and act.
    """
    def get_line(self):
        return self.rfile.readline().strip()

    def send_action(self, action):
        logger.info("Sending action: %d = %s", 
                    Action.qcog_action(action), 
                    Action.action_name(action))
        action = Action.qcog_action(action)
        self.wfile.write("{}\n".format(action))
        self.wfile.flush()


    def log_perception(self):
        logger.info("Perception: %s received for action: %s", 
                      str(self.perception), 
                      Action.action_name(self.action))

    def handle(self):
        logger.info("Handling pcog model learning connection request")
        logger.info("Connection address: {}".format(self.client_address[0]))
        self.send_action(Action.random_action())
        self.past = process(self.get_line())
        self.action = Action.random_action()
        self.send_action(self.action)
        self.recent = process(self.get_line())
        self.perception = perceive(self.recent, self.past)
        self.memory = UtileSuffixMemory()
        self.memory.insert(Instance(
            action=self.action,
            observation=self.perception,
            reward=perception_reward(self.recent, self.past),
        ))
        exploration_iterations = 1000
        logger.info("Starting Agent loop with %d iterations of exploration", exploration_iterations)
        while self.recent:
            self.log_perception()
            if False and exploration_iterations < 0:
                pass
            else:
                self.action = Action.random_action()
                self.send_action(self.action)
                exploration_iterations -= 1
            self.memory.insert(Instance(
                action=self.action,
                observation=self.perception,
                reward=perception_reward(self.recent, self.past),
            ))
            self.perception = perceive(self.recent, self.past)
            self.past = self.recent
            self.recent = process(self.get_line())

def main(args=None):
    HOST, PORT = "localhost", 9999
    server = SocketServer.TCPServer((HOST, PORT), PCogModelLearnerHandler)
    server.serve_forever()

if __name__=="__main__": 
    main(sys.argv[1:])
