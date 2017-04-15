from .dialog_manager_super import DialogManagerBase


class TelegramDialogManager(DialogManagerBase):

    def initialize_episode(self):
        """ Refresh state for new dialog """

        super().initialize_episode()
        self.user.initialize_episode()
        self.agent.initialize_episode()

    def next_turn(self, message):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################

        user_action = self.user.next(message)
        self.state_tracker.update(user_action=user_action)

        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################

        agent_state = self.state_tracker.get_state_for_agent()
        agent_action = self.agent.state_to_action(agent_state)

        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=agent_action)

        agent_ans = self.agent.add_nl_to_action(agent_action)

        if user_action['diaact'] == "thanks":
            agent_ans = 'Thank you, good bye!'
            self.episode_over = True

        return self.episode_over, agent_ans