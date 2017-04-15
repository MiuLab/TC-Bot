''' 

Base class for DialogManager

'''


class DialogManagerBase:
    """ A dialog manager to mediate the interaction between an agent and a customer """

    def __init__(self, agent, user, state_tracker):
        self.agent = agent
        self.user = user
        self.state_tracker = state_tracker
        self.user_action = None
        self.episode_over = False

    def initialize_episode(self):
        """ Refresh state for new dialog """

        self.episode_over = False
        self.state_tracker.initialize_episode()


    def next_turn(self, **kwargs):
        """ This function initiates each subsequent
         exchange between agent and user """
        pass
