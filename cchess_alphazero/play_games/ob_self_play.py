import os
import sys
import subprocess
import pygame
import copy
import numpy as np
from collections import defaultdict
from logging import getLogger
from time import sleep, time
from datetime import datetime
from pygame.locals import *
from threading import Thread

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.play_games.ob_self_play_helpers import load_image, load_images, Chessman_Sprite, creat_sprite_group, select_sprite_from_group

logger = getLogger(__name__)
main_dir = os.path.split(os.path.abspath(__file__))[0]
PIECE_STYLE = 'WOOD'

def start(config: Config, ucci=False, ai_move_first=True, gui=False):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    if not ucci:
        if gui:
            play = ObSelfPlayGUI(config)
        else:
            play = ObSelfPlay(config)
    else:
        play = ObSelfPlayUCCI(config, ai_move_first)
    play.start()

class ObSelfPlay:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()
        history = [self.env.get_state()]

        while not self.env.board.is_end():
            no_act = None
            state = self.env.get_state()
            if state in history[:-1]:
                no_act = []
                for i in range(len(history) - 1):
                    if history[i] == state:
                        no_act.append(history[i + 1])
            action, _ = self.ai.action(state, self.env.num_halfmoves, no_act)
            history.append(action)
            if action is None:
                print("AI投降了!")
                break
            move = self.env.board.make_single_record(int(action[0]), int(action[1]), int(action[2]), int(action[3]))
            if not self.env.red_to_move:
                action = flip_move(action)
            self.env.step(action)
            history.append(self.env.get_state())
            print(f"AI选择移动 {move}")
            self.env.board.print_to_cl()
            sleep(1)

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()

class ObSelfPlayUCCI:
    def __init__(self, config: Config, ai_move_first=True):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.ai_move_first = ai_move_first

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def start(self):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        self.env.board.print_to_cl()
        history = [self.env.get_state()]
        turns = 0
        game_over = False
        final_move = None

        while not game_over:
            if (self.ai_move_first and turns % 2 == 0) or (not self.ai_move_first and turns % 2 == 1):
                start_time = time()
                no_act = None
                state = self.env.get_state()
                if state in history[:-1]:
                    no_act = []
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            act = history[i + 1]
                            if not self.env.red_to_move:
                                act = flip_move(act)
                            no_act.append(act)
                action, _ = self.ai.action(state, self.env.num_halfmoves, no_act)
                end_time = time()
                if action is None:
                    print("AlphaZero 投降了!")
                    break
                move = self.env.board.make_single_record(int(action[0]), int(action[1]), int(action[2]), int(action[3]))
                print(f"AlphaZero 选择移动 {move}, 消耗时间 {(end_time - start_time):.2f}s")
                if not self.env.red_to_move:
                    action = flip_move(action)
            else:
                state = self.env.get_state()
                print(state)
                fen = senv.state_to_fen(state, turns)
                action = self.get_ucci_move(fen)
                if action is None:
                    print("Eleeye 投降了!")
                    break
                print(action)
                if not self.env.red_to_move:
                    rec_action = flip_move(action)
                else:
                    rec_action = action
                move = self.env.board.make_single_record(int(rec_action[0]), int(rec_action[1]), int(rec_action[2]), int(rec_action[3]))
                print(f"Eleeye 选择移动 {move}")
            history.append(action)
            self.env.step(action)
            history.append(self.env.get_state())
            self.env.board.print_to_cl()
            turns += 1
            sleep(1)
            game_over, final_move = self.env.board.is_end_final_move()
            print(game_over, final_move)

        if final_move:
            move = self.env.board.make_single_record(int(final_move[0]), int(final_move[1]), int(final_move[2]), int(final_move[3]))
            print(f"Final Move {move}")
            if not self.env.red_to_move:
                final_move = flip_move(final_move)
            self.env.step(final_move)
            self.env.board.print_to_cl()

        self.ai.close()
        print(f"胜者是 is {self.env.board.winner} !!!")
        self.env.board.print_record()

    def get_ucci_move(self, fen, time=3):
        p = subprocess.Popen(self.config.resource.eleeye_path,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
        setfen = f'position fen {fen}\n'
        setrandom = 'setoption randomness small\n'
        cmd = 'ucci\n' + setrandom + setfen + f'go time {time * 1000}\n'
        try:
            out, err = p.communicate(cmd, timeout=time+0.5)
        except:
            p.kill()
            try:
                out, err = p.communicate()
            except Exception as e:
                logger.error(f"{e}, cmd = {cmd}")
                return self.get_ucci_move(fen, time+1)
        print(out)
        lines = out.split('\n')
        if lines[-2] == 'nobestmove':
            return None
        move = lines[-2].split(' ')[1]
        if move == 'depth':
            move = lines[-1].split(' ')[6]
        return senv.parse_ucci_move(move)


class ObSelfPlayGUI:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.winstyle = 0
        self.screen_width = 720
        self.height = 577
        self.width = 521
        self.chessman_w = 57
        self.chessman_h = 57
        self.disp_record_num = 15
        self.rec_labels = [None] * self.disp_record_num
        self.nn_value = 0
        self.mcts_moves = {}
        self.history = []
        if self.config.opts.bg_style == 'WOOD':
            self.chessman_w += 1
            self.chessman_h += 1

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def init_screen(self):
        bestdepth = pygame.display.mode_ok([self.screen_width, self.height], self.winstyle, 32)
        screen = pygame.display.set_mode([self.screen_width, self.height], self.winstyle, bestdepth)
        pygame.display.set_caption("中国象棋 Zero - AI 自对弈观察")
        bgdtile = load_image(f'{self.config.opts.bg_style}.GIF')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.height])
        board_background.blit(bgdtile, (0, 0))
        widget_background = pygame.Surface([self.screen_width - self.width, self.height])
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)

        font_file = self.config.resource.font_path
        if os.path.exists(font_file):
            font = pygame.font.Font(font_file, 16)
        else:
            font = pygame.font.SysFont('simsun', 16)
        font_color = (0, 0, 0)
        font_background = (255, 255, 255)
        t = font.render("着法记录", True, font_color, font_background)
        t_rect = t.get_rect()
        t_rect.x = 10
        t_rect.y = 10
        widget_background.blit(t, t_rect)

        screen.blit(board_background, (0, 0))
        screen.blit(widget_background, (self.width, 0))
        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w, self.chessman_h)
        return screen, board_background, widget_background

    def start(self):
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=True)

        pygame.init()
        screen, board_background, widget_background = self.init_screen()
        framerate = pygame.time.Clock()

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        ai_worker = Thread(target=self.ai_move_loop, name="ai_worker")
        ai_worker.daemon = True
        ai_worker.start()

        while not self.env.board.is_end():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.board.print_record()
                    self.ai.close(wait=False)
                    game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
                    self.env.board.save_record(path)
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    pass

            self.draw_widget(screen, widget_background)
            framerate.tick(20)
            self.chessmans.clear(screen, board_background)
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()

        self.ai.close(wait=False)
        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()
        
        # Display winner information clearly
        print("\n" + "="*50)
        if self.env.board.winner == Winner.red:
            print("🎉 红方获胜！")
        elif self.env.board.winner == Winner.black:
            print("🎉 黑方获胜！")
        else:
            print("🤝 和棋！")
        print("="*50 + "\n")
        
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(path)
        sleep(3)

    def ai_move_loop(self):
        self.history = [self.env.get_state()]
        no_act = None
        while not self.env.done:
            labels = ActionLabelsRed
            labels_n = len(ActionLabelsRed)
            self.ai.search_results = {}
            state = self.env.get_state()
            logger.info(f"state = {state}")
            _, _, _, check = senv.done(state, need_check=True)
            if not check and state in self.history[:-1]:
                no_act = []
                free_move = defaultdict(int)
                for i in range(len(self.history) - 1):
                    if self.history[i] == state:
                        if senv.will_check_or_catch(state, self.history[i+1]):
                            no_act.append(self.history[i + 1])
                        else:
                            free_move[state] += 1
                            if free_move[state] >= 2:
                                self.env.winner = Winner.draw
                                self.env.board.winner = Winner.draw
                                break
                if no_act:
                    logger.debug(f"no_act = {no_act}")
            action, policy = self.ai.action(state, self.env.num_halfmoves, no_act)
            if action is None:
                logger.info("AI has resigned!")
                return
            self.history.append(action)
            if not self.env.red_to_move:
                action = flip_move(action)
            key = self.env.get_state()
            p, v = self.ai.debug[key]
            logger.info(f"check = {check}, NN value = {v:.3f}")
            self.nn_value = v
            logger.info("MCTS results:")
            self.mcts_moves = {}
            for move, action_state in self.ai.search_results.items():
                move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                logger.info(f"move: {move_cn}-{move}, visit count: {action_state[0]}, Q_value: {action_state[1]:.3f}, Prior: {action_state[2]:.3f}")
                self.mcts_moves[move_cn] = action_state
            x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
            logger.info(f"AI move: ({x0},{y0}) -> ({x1},{y1})")
            chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
            sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
            if sprite_dest:
                self.chessmans.remove(sprite_dest)
                sprite_dest.kill()
            if chessman_sprite:
                chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
                logger.info(f"Chessman sprite moved successfully")
            else:
                logger.error(f"Cannot find chessman sprite at ({x0},{y0})")
            self.history.append(self.env.get_state())

    def draw_widget(self, screen, widget_background):
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)
        pygame.draw.line(widget_background, (255, 0, 0), (10, 285), (self.screen_width - self.width - 10, 285))
        screen.blit(widget_background, (self.width, 0))
        self.draw_records(screen, widget_background)
        self.draw_evaluation(screen, widget_background)

    def draw_records(self, screen, widget_background):
        text = '着法记录'
        self.draw_label(screen, widget_background, text, 10, 16, 10)
        records = self.env.board.record.split('\n')
        font_file = self.config.resource.font_path
        if os.path.exists(font_file):
            font = pygame.font.Font(font_file, 12)
        else:
            font = pygame.font.SysFont('simsun', 12)
        i = 0
        for record in records[-self.disp_record_num:]:
            self.rec_labels[i] = font.render(record, True, (0, 0, 0), (255, 255, 255))
            t_rect = self.rec_labels[i].get_rect()
            t_rect.y = 35 + i * 15
            t_rect.x = 10
            t_rect.width = self.screen_width - self.width
            widget_background.blit(self.rec_labels[i], t_rect)
            i += 1
        screen.blit(widget_background, (self.width, 0))

    def draw_evaluation(self, screen, widget_background):
        title_label = 'CC-Zero 信息'
        self.draw_label(screen, widget_background, title_label, 300, 16, 10)
        info_label = f'MCTS 搜索次数：{self.config.play.simulation_num_per_move}'
        self.draw_label(screen, widget_background, info_label, 335, 14, 10)
        eval_label = f"当前局势评估：{self.nn_value:.3f}"
        self.draw_label(screen, widget_background, eval_label, 360, 14, 10)
        label = f"MCTS 搜索结果:"
        self.draw_label(screen, widget_background, label, 395, 14, 10)
        label = f"着法 访问计数 动作价值 先验概率"
        self.draw_label(screen, widget_background, label, 415, 12, 10)
        i = 0
        tmp = copy.deepcopy(self.mcts_moves)
        for mov, action_state in tmp.items():
            label = f"{mov}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 10)
            label = f"{action_state[0]}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 70)
            label = f"{action_state[1]:.2f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 100)
            label = f"{action_state[2]:.3f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 150)
            i += 1

    def draw_label(self, screen, widget_background, text, y, font_size, x=None):
        font_file = self.config.resource.font_path
        if os.path.exists(font_file):
            font = pygame.font.Font(font_file, font_size)
        else:
            font = pygame.font.SysFont('simsun', font_size)
        label = font.render(text, True, (0, 0, 0), (255, 255, 255))
        t_rect = label.get_rect()
        t_rect.y = y
        if x != None:
            t_rect.x = x
        else:
            t_rect.centerx = (self.screen_width - self.width) / 2
        widget_background.blit(label, t_rect)
        screen.blit(widget_background, (self.width, 0))
