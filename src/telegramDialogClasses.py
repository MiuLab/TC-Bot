from deep_dialog.dialog_system.state_tracker import StateTracker

class TelegramDialogManager():
    def __init__(self, agent, user, act_set, slot_set, movie_dictionary):
        self.agent = agent
        self.user = user
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.episode_over = False

    def initialize_episode(self, message):
        """ Refresh state for new dialog """
        """ :param message: text said by user in Telegram for the first time"""

        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action = self.user.initialize_episode(message)
        self.state_tracker.update(user_action=self.user_action)
        self.agent.initialize_episode()

    def next_turn(self, message):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################
        self.state = self.state_tracker.get_state_for_agent()
        self.agent_action = self.agent.state_to_action(self.state)

        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)

        agent_ans = self.agent.add_nl_to_action(self.agent_action)

        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action, message)
        if self.user_action['diaact'] == "thanks":
            agent_ans = 'Thank you, good bye!'
            self.episode_over = True

        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if not self.episode_over:
            self.state_tracker.update(user_action=self.user_action)

        ########################################################################
        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        ########################################################################
        return self.episode_over, agent_ans