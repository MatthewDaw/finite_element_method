"""Class for determining when to switchtraining strategies."""

class PolicySwitcher:
    
    def __init__(self, threshold_to_switch):
        self.success_num = 0 
        self.threshold_to_switch = threshold_to_switch
        self.in_proceeding_phase = False 
        
    def add_success(self):
        if self.success_num < self.threshold_to_switch + 10:
            self.success_num += 1

    def add_failure(self):
        if self.success_num > -10:
            self.success_num -= 1

    def set_massive_failure(self):
        self.in_proceeding_phase = False
        self.success_num = 0

    def check_if_ready_to_proceed(self):
        if self.success_num >= self.threshold_to_switch:
            self.in_proceeding_phase = True
            return True 
        if self.in_proceeding_phase and self.success_num > 0:
            return True 
        return False
