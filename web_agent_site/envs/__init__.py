from gym.envs.registration import register

from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from web_agent_site.envs.web_agent_image_env import WebAgentImageEnv

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)

register(
  id='WebAgentImageEnv-v0',
  entry_point='web_agent_site.envs:WebAgentImageEnv',
)
