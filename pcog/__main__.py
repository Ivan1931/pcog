"""
Main entry point for PCog. Can open two types of agents - a model learning agent
or a hand crafted POMPD based agent
"""
import sys
import logging
import SocketServer
import datetime
from json import loads
from bunch import bunchify
from .agent import GridAgent
from .envconf import Action
from .perception import process
from .usm import UtileSuffixMemory
from .model_learn_agent import ModelLearnAgent

logging.basicConfig(filename="pcog.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Session begins: {}".format(datetime.datetime.now()))


class PCogHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        logger.info("Handling pcog connection request")
        self.data = self.rfile.readline().strip()
        agent = GridAgent()
        agent.derive_model()
        while self.data:
            logger.info("{} wrote:".format(self.client_address[0]))
            logger.info(self.data)
            observation = agent.derive_observation(bunchify(loads(self.data)))
            agent.update_belief(observation)
            action = agent.plan()
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

    def handle(self):
        logger.info("Handling pcog model learning connection request")
        logger.info("Connection address: {}".format(self.client_address[0]))
        self.agent = ModelLearnAgent(usm=UtileSuffixMemory(
            known_actions=list(Action.SET),
        ))
        self.recent = self.get_line()
        self.send_action(self.agent.get_decision())
        self.recent = self.get_line()
        logger.info("Starting agent exploration loop")
        while self.recent:
            raw_perception = process(self.recent)
            self.agent.add_perception(
                raw_perception
            )
            self.send_action(self.agent.get_decision())
            logger.info("=" * 20)
            self.recent = self.get_line()

class PCogModelLearnerEvaluatorHandler(PCogModelLearnerHandler):
    def handle(self):
        logger.info("Starting experiment on PCog Model Learner")
        logger.info("Connection address: {}".format(self.client_address[0]))
        self.agent = ModelLearnAgent(usm=UtileSuffixMemory(
            known_actions=list(Action.SET)
        ))
        self.recent = self.get_line()
        self.send_action(self.agent.get_decision())
        self.recent = self.get_line()
        logger.info("Starting agent exploration loop")
        while self.recent:
            if self.recent.strip() != "RESTART":
                raw_perception = process(self.recent)
                self.agent.add_perception(raw_perception)
                self.send_action(self.agent.get_decision())
                logger.info("=" * 20)
            else:
                logger.info("====== ITERATION RESTARTED ======")
                self.agent = ModelLearnAgent(usm=UtileSuffixMemory(
                    known_actions=list(Action.SET)
                ))
            self.recent = self.get_line()


def main(args=None):
    HOST, PORT = "localhost", 9999
    server = SocketServer.TCPServer((HOST, PORT), PCogModelLearnerHandler)
    server.serve_forever()

if __name__=="__main__": 
    main(sys.argv[1:])
