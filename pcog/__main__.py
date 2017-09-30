"""
Main entry point for PCog. Can open two types of agents - a model learning agent
or a hand crafted POMPD based agent
"""
import sys
import logging
import SocketServer
import datetime
import threading
from argparse import ArgumentParser
from json import loads
from bunch import bunchify
from random import choice, random
from .agent import GridAgent, run_skinny_pomdp
from .envconf import Action
from .perception import process
from .usm import UtileSuffixMemory
from .model_learn_agent import ModelLearnAgent

logging.basicConfig(filename="pcog.log", filemode="w", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Session begins: {}".format(datetime.datetime.now()))

def last_n_same(lst, n):
    xs = lst[-n:]
    if len(xs) == 0:
        return True
    else:
        return all(map(lambda i: i == xs[0], xs))

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


class SkinnyPCogHandler(PCogModelLearnerHandler):
    def send_raw_action(self, action):
        logger.info("Sending action: %d = %s", action, Action.action_name(action))
        self.wfile.write("{}\n".format(action))
        self.wfile.flush()

    def handle(self):
        logger.info("Starting experiment with Skinny PCog model learner")
        logger.info("Connection address: {}".format(self.client_address[0]))
        self.recent = self.get_line()
        actions = []
        epsilon = 0.05
        while self.recent:
            pomdp_data = bunchify(loads(self.recent))
            if last_n_same(actions, 20) or random() < epsilon:
                logger.info("Choosing random action because we're stuck in a loop or we're feeling lucky")
                action = choice(list(Action.SET))
            else:
                logger.info("Choosing action with belief State: %s", " ".join(map(str, pomdp_data.beliefState)))
                action = run_skinny_pomdp(pomdp_data)
            actions.append(action)
            self.send_raw_action(action)
            self.recent = self.get_line()


class ThreadedServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    pass


def main():
    parser = ArgumentParser()
    parser.add_argument("--skinny", help="tells pcog to interpret a POMDP off the wire", action="store_true")
    args = parser.parse_args()

    HOST, PORT = "localhost", 9999
    if args.skinny:
        server = ThreadedServer((HOST, PORT), SkinnyPCogHandler)
    else:
        server = ThreadedServer((HOST, PORT), PCogModelLearnerHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    while server_thread.isAlive():
        pass


if __name__=="__main__": 
    main()
