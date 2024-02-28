import gym
import json
import random
import string
import time
import torch

from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import defaultdict
from flask import Flask
from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    ACTION_TO_TEMPLATE,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
from web_agent_site.engine.goal import get_reward, get_goals
from web_agent_site.utils import (
    DEFAULT_FILE_PATH,
    FEAT_CONV,
    FEAT_IDS,
    random_idx
)
import numpy as np
import web_agent_site.local_utils as local_utils
app = Flask(__name__)
class WebAgentImageEnv(gym.Env):
    """Gym environment for Text mode of WebShop environment"""
    def __init__(
            self,
            observation_mode='image',
            file_path=DEFAULT_FILE_PATH,
            server=None,
            **kwargs
        ):
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(WebAgentImageEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        self.file_path = file_path

        self.base_url = 'http://127.0.0.1:3000'
        self.server = SimServer(
            self.base_url,
            self.file_path,
            self.kwargs.get('filter_goals'),
            self.kwargs.get('limit_goals', -1),
            self.kwargs.get('num_products'),
            self.kwargs.get('human_goals'),
            self.kwargs.get('show_attrs', False),
        ) if server is None else server
        self.browser = SimBrowser(self.server)

        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')
        if self.kwargs.get('get_image', 0):
            self.feats = torch.load(FEAT_CONV) #torch.Size([2449321, 512])
            self.ids = torch.load(FEAT_IDS) #a list of 2449321 urls
            self.ids = {url: idx for idx, url in enumerate(self.ids)} #a dict of url:index for 2449321 images
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        self.reset()

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        info = None
        self.get_available_actions()

        # Determine action type (click, search) and argument
        action_name, action_arg = parse_action(action)
        if action_arg is not None:
            action_arg = action_arg.lower()
        if (action_name == 'search' and 
            action_arg is not None and 
            action_arg != ''):
            status = self.browser.search(action_arg)
        elif (action_name == 'click' and 
              action_arg in self.text_to_clickable.keys() and 
              action_arg != 'search'):
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:
            status = dict(reward=0, done=False)

        # Update observation, state with the new action
        ob = self.observation
        if self.observation_mode == 'image':
            ob_list = [ob]
            act_list = []
            self.prev_actions.append(action)
            for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
                if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                    act_list.append(self.prev_actions[-i])
                if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                    ob_list.append(self.prev_obs[-i])
            self.prev_obs.append(ob)
            state = ( ' [SEP] '.join(text_list[::-1]) , ob_list )
            return state, status['reward'], status['done'], info
        else:
            text_list = [ob]
            self.prev_actions.append(action)
            for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
                if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                    text_list.append(self.prev_actions[-i])
                if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                    text_list.append(self.prev_obs[-i])
            state = ' [SEP] '.join(text_list[::-1])
            self.prev_obs.append(ob)
            return state, status['reward'], status['done'], info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links  = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f'{b.get_text()}'.lower(): b
            for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get('value')
            self.text_to_clickable[f'{opt_value}'] = opt
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )
    
    def get_image(self): #### NOTE: see what this does, i.e. what is URL etc.
        """Scrape image from page HTML and return as a list of pixel values"""
        # html_obj = self._parse_html(self.browser.page_source)
        # image_url = html_obj.find(id='product-image')
        # if image_url is not None:
        #     image_url = image_url['src']
        #     if image_url in self.ids:
        #         image_idx = self.ids[image_url]
        #         image = self.feats[image_idx]
        #         return image
        url = self.state['url']

        from selenium import webdriver
        from PIL import Image
        driver = webdriver.Chrome()
        driver.get(url)
        driver.save_screenshot("temp_image.png")
        image = Image.open("temp_image.png").convert("RGB")
        image_url = "temp_image.png"
        from local_utils import Preprocess, Config, GeneralizedRCNN
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        image_preprocess = Preprocess(frcnn_cfg)
        images, sizes, scales_yx = image_preprocess(image_url)
        frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
        output_dict = frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        features = output_dict.get("roi_features") # [1, 36, 2048])
        
        return features
        #return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj
    
    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        elif self.observation_mode =='image':
            return self.get_image()
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )
    
    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )
    
    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            for t in visible_texts:
                if t == '\n': continue
                if t.parent.name == 'button':  # button
                    processed_t = f'[button] {t} [button_]'
                elif t.parent.name == 'label':  # options
                    if f'"{t}"' in self.state['url']:
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'  [button] {t} [button_]'
                elif t.parent.get('class') == ["product-link"]: # product asins
                    if f'{t}' in self.server.user_sessions[self.session]['asins']:
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    else:
                        processed_t = f'\n[button] {t} [button_]'
                else: # regular, unclickable text
                    processed_t =  str(t)
                observation += processed_t + '\n'
            return observation
    
    def reset(self, session=None, instruction_text=None):
        """Create a new session and reset environment variables"""
        session_int = None
        if session is not None:
            self.session = str(session)
            if isinstance(session, int):
                session_int = session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=10))
        if self.session_prefix is not None:
            self.session = self.session_prefix + self.session

        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        self.text_to_clickable = None
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []
        return obs, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


class SimServer:
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""
    def __init__(
        self,
        base_url,
        file_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
    ):
        """
        Constructor for simulated server serving WebShop application
        
        Arguments:
        filter_goals (`func`) -- Select specific goal(s) for consideration based on criteria of custom function
        limit_goals (`int`) -- Limit to number of goals available
        num_products (`int`) -- Number of products to search across
        human_goals (`bool`) -- If true, load human goals; otherwise, load synthetic goals
        """
        # Load all products, goals, and search engine
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices, _ = \
            load_products(filepath=file_path, num_products=num_products, human_goals=human_goals)
        self.search_engine = init_search_engine(num_products=num_products)
        self.goals = get_goals(self.all_products, self.product_prices, human_goals)
        self.show_attrs = show_attrs

        # Fix outcome for random shuffling of goals
        random.seed(233)
        random.shuffle(self.goals)

        # Apply `filter_goals` parameter if exists to select speific goal(s)
        if filter_goals is not None:
            self.goals = [
                goal for (i, goal) in enumerate(self.goals)
                if filter_goals(i, goal)
            ]
        
        # Imposes `limit` on goals via random selection
        if limit_goals != -1 and limit_goals < len(self.goals):
            self.weights = [goal['weight'] for goal in self.goals]
            self.cum_weights = [0]
            for w in self.weights:
                self.cum_weights.append(self.cum_weights[-1] + w)
            idxs = []
            while len(idxs) < limit_goals:
                idx = random_idx(self.cum_weights)
                if idx not in idxs:
                    idxs.append(idx)
            self.goals = [self.goals[i] for i in idxs]
        print(f'Loaded {len(self.goals)} goals.')

        # Set extraneous housekeeping variables
        self.weights = [goal['weight'] for goal in self.goals]
        self.cum_weights = [0]
        for w in self.weights:
            self.cum_weights.append(self.cum_weights[-1] + w)
        self.user_sessions = dict()
        self.search_time = 0
        self.render_time = 0
        self.sample_time = 0
        self.assigned_instruction_text = None  # TODO: very hacky, should remove
        
    @app.route('/', methods=['GET', 'POST'])
    def index(self, session_id, **kwargs):
        """Redirect to the search page with the given session ID"""
        html = map_action_to_html(
            'start',
            session_id=session_id,
            instruction_text=kwargs['instruction_text'],
        )
        url = f'{self.base_url}/{session_id}'
        return html, url
    
    @app.route('/', methods=['GET', 'POST'])
    def search_results(self, session_id, **kwargs):
        """Initialize session and return the search results page"""
        session = self.user_sessions[session_id]
        keywords = kwargs['keywords']  # TODO: why is this using kwargs? why not session?
        assert isinstance(keywords, list)
        page = 1 if 'page' not in kwargs else kwargs['page']
        session["page"] = page
        session["keywords"] = keywords
        session["actions"]["search"] += 1
        session["asin"] = None
        session["options"] = {}

        # Perform search on keywords from items and record amount of time it takes
        old_time = time.time()
        top_n_products = get_top_n_product_from_keywords(
            keywords,
            self.search_engine,
            self.all_products,
            self.product_item_dict,
        )
        self.search_time += time.time() - old_time
        
        # Get product list from search result asins and get list of corresponding URLs
        products = get_product_per_page(top_n_products, page)

        keywords_url_string = '+'.join(keywords)
        url = (
            f'{self.base_url}/search_results/{session_id}/'
            f'{keywords_url_string}/{page}'
        )

        # Render HTML search page and record amount of time taken
        old_time = time.time()
        html = map_action_to_html(
            'search',
            session_id=session_id,
            products=products,
            keywords=session["keywords"],
            page=page,
            total=len(top_n_products),
            instruction_text=session["goal"]["instruction_text"],
        )
        self.render_time += time.time() - old_time
        return html, url
    
    @app.route('/', methods=['GET', 'POST'])
    def item_page(self, session_id, **kwargs):
        """Render and return the HTML for a product item page"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        text_to_clickable = kwargs['text_to_clickable']
        clickable = text_to_clickable[clickable_name]

        # Update session logs with information of last product asin selected
        if (clickable.get('class') is not None and
            clickable.get('class')[0] == 'product-link'):
            session["asin"] = clickable_name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif clickable.get('name') is not None:
            clickable_key = clickable['name'].lower()
            session["options"][clickable_key] = clickable_name
            session["actions"]["options"] += 1

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        keywords_url_string = '+'.join(session["keywords"])
        option_string = json.dumps(session['options'])

        url = (
            f'{self.base_url}/item_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/'
            f'{session["page"]}/{option_string}'
        )

        html = map_action_to_html(
            'click',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
            show_attrs=self.show_attrs,
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_sub_page(self, session_id, **kwargs):
        """Render and return the HTML for a product's sub page (i.e. description, features)"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break
        
        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        session["actions"][clickable_name] += 1
        keywords_url_string = '+'.join(session["keywords"])
        url = (
            f'{self.base_url}/item_sub_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/{session["page"]}/'
            f'{clickable_name}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{clickable_name}]',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def done(self, session_id, **kwargs):
        """Render and return HTML for done page"""
        session = self.user_sessions[session_id]
        goal = self.user_sessions[session_id]['goal']
        purchased_product = self.product_item_dict[session["asin"]]
        session["actions"]["purchase"] += 1
        price = self.product_prices.get(session["asin"])

        # Calculate reward for selected product and set variables for page details
        reward, info = get_reward(
            purchased_product,
            goal,
            price=price,
            options=session["options"],
            verbose=True
        )

        self.user_sessions[session_id]['verbose_info'] = info
        self.user_sessions[session_id]['done'] = True
        self.user_sessions[session_id]['reward'] = reward

        url = (
            f'{self.base_url}/done/{session_id}/'
            f'{session["asin"]}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{END_BUTTON}]',
            session_id=session_id,
            reward=reward,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url, reward
    
    def receive(self, session_id, current_url, session_int=None, **kwargs):
        """Map action to the corresponding page"""
        status = dict(reward=0.0, done=False)

        with app.app_context(), app.test_request_context():
            # Create/determine goal, instruction_text from current session
            if session_id not in self.user_sessions:
                idx = session_int if (session_int is not None and isinstance(session_int, int)) else random_idx(self.cum_weights) 
                goal = self.goals[idx]
                instruction_text = goal['instruction_text']
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            else:
                instruction_text = \
                    self.user_sessions[session_id]['goal']['instruction_text']
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text  # TODO: very hacky, should remove
                self.user_sessions[session_id]['goal']['instruction_text'] = instruction_text
            session = self.user_sessions[session_id]

            if not kwargs:
                # If no action, reset the session variables
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update(
                    {
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'options': dict(),
                        'actions': defaultdict(int)
                    }
                )
            elif 'keywords' in kwargs:
                # If search keywords are available, run a search
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    # If "buy now" clicked, calculate reward and flag session as terminated
                    html, url, reward = self.done(session_id, **kwargs)
                    status['reward'] = reward
                    status['done'] = True
                elif clickable_name == BACK_TO_SEARCH.lower():
                    # If "back to search" clicked, recursively reset the session back to search page
                    html, url, status = self.receive(session_id, current_url)
                elif (clickable_name == NEXT_PAGE.lower() and 
                      self.get_page_name(current_url) == 'search_results'):
                    # If "next page" clicked from search results, re-render with `page` enumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] + 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and 
                      self.get_page_name(current_url) == 'search_results'):
                    # If "prev page" clicked from search results, re-render with `page` denumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] - 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and 
                      self.get_page_name(current_url) == 'item_sub_page'):
                    # If "prev page" clicked from sub page, return to corresponding item page
                    html, url = self.item_page(session_id, **kwargs)
                elif (clickable_name == PREV_PAGE.lower() and 
                      self.get_page_name(current_url) == 'item_page'):
                    # If "prev page" clicked from item page, return to search results page
                    html, url = self.search_results(
                        session_id,
                        keywords=session["keywords"],
                        page=session["page"],
                        **kwargs
                    )
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    # Render item_sub_page if clickable is description, features, or reviews
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    # Otherwise, render current item page
                    html, url = self.item_page(session_id, **kwargs)
            return html, url, status
    
    def get_page_name(self, url):
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        if url is None:
            return None
        page_names = [
            'search_results',
            'item_page',
            'item_sub_page',
            'done'
        ]
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''  # index page


class SimBrowser:
    """Simulated browser for rendering the HTML source of WebShop environment pages"""
    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(self, url, session_id=None, session_int=None):
        """Set browser variables to corresponding link, page HTML for URL"""
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        self.page_source, _, _ = \
            self.server.receive(self.session_id, self.current_url, session_int=session_int)
        self.current_url = url
    
    def click(self, clickable_name, text_to_clickable):
        """Wrapper for `receive` handler for performing click action on current page"""
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                clickable_name=clickable_name,
                text_to_clickable=text_to_clickable,
            )
        return status
    
    def search(self, keywords):
        """Wrapper for `receive` handler for performing search action on current page"""
        if isinstance(keywords, str):
            keywords = keywords.split(' ')
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                keywords=keywords,
        )
        return status


class SingleImageViz:
    def __init__(
        self,
        img,
        scale=1.2,
        edgecolor="g",
        alpha=0.5,
        linestyle="-",
        saveas="test_out.jpg",
        rgb=True,
        pynb=False,
        id2obj=None,
        id2attr=None,
        pad=0.7,
    ):
        """
        img: an RGB image of shape (H, W, 3).
        """
        if isinstance(img, torch.Tensor):
            img = img.numpy().astype("np.uint8")
        if isinstance(img, str):
            img = img_tensorize(img)
        assert isinstance(img, np.ndarray)

        width, height = img.shape[1], img.shape[0]
        fig = mplfigure.Figure(frameon=False)
        dpi = fig.get_dpi()
        width_in = (width * scale + 1e-2) / dpi
        height_in = (height * scale + 1e-2) / dpi
        fig.set_size_inches(width_in, height_in)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, width)
        ax.set_ylim(height)

        self.saveas = saveas
        self.rgb = rgb
        self.pynb = pynb
        self.img = img
        self.edgecolor = edgecolor
        self.alpha = 0.5
        self.linestyle = linestyle
        self.font_size = int(np.sqrt(min(height, width)) * scale // 3)
        self.width = width
        self.height = height
        self.scale = scale
        self.fig = fig
        self.ax = ax
        self.pad = pad
        self.id2obj = id2obj
        self.id2attr = id2attr
        self.canvas = FigureCanvasAgg(fig)

    def add_box(self, box, color=None):
        if color is None:
            color = self.edgecolor
        (x0, y0, x1, y1) = box
        width = x1 - x0
        height = y1 - y0
        self.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=self.font_size // 3,
                alpha=self.alpha,
                linestyle=self.linestyle,
            )
        )

    def draw_boxes(self, boxes, obj_ids=None, obj_scores=None, attr_ids=None, attr_scores=None):
        if len(boxes.shape) > 2:
            boxes = boxes[0]
        if len(obj_ids.shape) > 1:
            obj_ids = obj_ids[0]
        if len(obj_scores.shape) > 1:
            obj_scores = obj_scores[0]
        if len(attr_ids.shape) > 1:
            attr_ids = attr_ids[0]
        if len(attr_scores.shape) > 1:
            attr_scores = attr_scores[0]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        assert isinstance(boxes, np.ndarray)
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        sorted_idxs = np.argsort(-areas).tolist()
        boxes = boxes[sorted_idxs] if boxes is not None else None
        obj_ids = obj_ids[sorted_idxs] if obj_ids is not None else None
        obj_scores = obj_scores[sorted_idxs] if obj_scores is not None else None
        attr_ids = attr_ids[sorted_idxs] if attr_ids is not None else None
        attr_scores = attr_scores[sorted_idxs] if attr_scores is not None else None

        assigned_colors = [self._random_color(maximum=1) for _ in range(len(boxes))]
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
        if obj_ids is not None:
            labels = self._create_text_labels_attr(obj_ids, obj_scores, attr_ids, attr_scores)
            for i in range(len(boxes)):
                color = assigned_colors[i]
                self.add_box(boxes[i], color)
                self.draw_labels(labels[i], boxes[i], color)

    def draw_labels(self, label, box, color):
        x0, y0, x1, y1 = box
        text_pos = (x0, y0)
        instance_area = (y1 - y0) * (x1 - x0)
        small = _SMALL_OBJ * self.scale
        if instance_area < small or y1 - y0 < 40 * self.scale:
            if y1 >= self.height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)

        height_ratio = (y1 - y0) / np.sqrt(self.height * self.width)
        lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
        font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
        font_size *= 0.75 * self.font_size

        self.draw_text(
            text=label,
            position=text_pos,
            color=lighter_color,
        )

    def draw_text(
        self,
        text,
        position,
        color="g",
        ha="left",
    ):
        rotation = 0
        font_size = self.font_size
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        bbox = {
            "facecolor": "black",
            "alpha": self.alpha,
            "pad": self.pad,
            "edgecolor": "none",
        }
        x, y = position
        self.ax.text(
            x,
            y,
            text,
            size=font_size * self.scale,
            family="sans-serif",
            bbox=bbox,
            verticalalignment="top",
            horizontalalignment=ha,
            color=color,
            zorder=10,
            rotation=rotation,
        )

    def save(self, saveas=None):
        if saveas is None:
            saveas = self.saveas
        if saveas.lower().endswith(".jpg") or saveas.lower().endswith(".png"):
            cv2.imwrite(
                saveas,
                self._get_buffer()[:, :, ::-1],
            )
        else:
            self.fig.savefig(saveas)

    def _create_text_labels_attr(self, classes, scores, attr_classes, attr_scores):
        labels = [self.id2obj[i] for i in classes]
        attr_labels = [self.id2attr[i] for i in attr_classes]
        labels = [
            f"{label} {score:.2f} {attr} {attr_score:.2f}"
            for label, score, attr, attr_score in zip(labels, scores, attr_labels, attr_scores)
        ]
        return labels

    def _create_text_labels(self, classes, scores):
        labels = [self.id2obj[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(li, s * 100) for li, s in zip(labels, scores)]
        return labels

    def _random_color(self, maximum=255):
        idx = np.random.randint(0, len(_COLORS))
        ret = _COLORS[idx] * maximum
        if not self.rgb:
            ret = ret[::-1]
        return ret

    def _get_buffer(self):
        if not self.pynb:
            s, (width, height) = self.canvas.print_to_buffer()
            if (width, height) != (self.width, self.height):
                img = cv2.resize(self.img, (width, height))
            else:
                img = self.img
        else:
            buf = io.BytesIO()  # works for cairo backend
            self.canvas.print_rgba(buf)
            width, height = self.width, self.height
            s = buf.getvalue()
            img = self.img

        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne  # fuse them with numexpr

            visualized_image = ne.evaluate("img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)")
        except ImportError:
            alpha = alpha.astype("float32") / 255.0
            visualized_image = img * (1 - alpha) + rgb * alpha

        return visualized_image.astype("uint8")

    def _change_color_brightness(self, color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color


# Color map
_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
