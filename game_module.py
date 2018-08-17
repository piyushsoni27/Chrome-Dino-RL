import numpy as np
from PIL import ImageGrab, Image
import cv2
import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys



#path variables
game_url = "game/dino.html"
chrome_driver_path = "./chromedriver.exe"
#loss_file_path = "./objects/loss_df.csv"
#actions_file_path = "./objects/actions_df.csv"
#scores_file_path = "./objects/scores_df.csv"

class Game:
    
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(executable_path = chrome_driver_path,chrome_options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.set_window_size(200, 300)
        self._driver.get(os.path.abspath(game_url))
    
    def _isplaying(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    
    def _iscrashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    
    def _restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
        
        time.sleep(0.25)# no actions are possible 
                        # for 0.25 sec after game starts, 
                        # skip learning at this time and make the model wait
                        
    def _pressUp(self):    
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
        
    def _pressDown(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    
    def _get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)
        
    def _pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    
    def _resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    
    def _end(self):
        self._driver.close()
        
        
class Dino:
    
    def __init__(self):
        self._game = Game()
        self._jump()
        time.sleep(0.5)
        
    def _jump(self):
        self._game._pressUp()
        
    def _duck(self):
        self._game._pressDown()
        
    def _isRunning(self):
        return self._game._isPlaying()
    
    def _isCrashed(self):
        return self._game._isCrashed()
        

if __name__ == "__main__":
    dino = Dino()
    dino._jump()
    time.sleep(0.5)
    dino._jump()
