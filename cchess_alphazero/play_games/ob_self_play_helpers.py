# Helper functions for ObSelfPlayGUI (imported from play.py)
import os
import pygame
from pygame.locals import *
from cchess_alphazero.environment.chessman import Rook, Cannon, Knight, King, Elephant, Mandarin

PIECE_STYLE = 'WOOD'  # default piece style

def load_image(file, sub_dir=None):
    '''loads an image, prepares it for play'''
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    if sub_dir:
        file = os.path.join(main_dir, 'images', sub_dir, file)
    else:
        file = os.path.join(main_dir, 'images', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s' %
                         (file, pygame.get_error()))
    return surface.convert()

def load_images(*files):
    global PIECE_STYLE
    imgs = []
    for file in files:
        imgs.append(load_image(file, PIECE_STYLE))
    return imgs

class Chessman_Sprite(pygame.sprite.Sprite):
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman, w=80, h=80):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = [pygame.transform.scale(image, (w, h)) for image in images]
        self.image = self.images[0]
        self.rect = Rect(chessman.col_num * w, (9 - chessman.row_num) * h, w, h)

    def move(self, col_num, row_num, w=80, h=80):
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect = Rect(old_col_num * w, (9 - old_row_num) * h, w, h)
            self.rect.move_ip((col_num - old_col_num)
                              * w, (old_row_num - row_num) * h)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            self.image = self.images[1]
        else:
            self.image = self.images[0]


def creat_sprite_group(sprite_group, chessmans_hash, w, h):
    for chess in chessmans_hash.values():
        if chess.is_red:
            if isinstance(chess, Rook):
                images = load_images("RR.GIF", "RRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("RC.GIF", "RCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("RN.GIF", "RNS.GIF")
            elif isinstance(chess, King):
                images = load_images("RK.GIF", "RKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("RB.GIF", "RBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.GIF", "RAS.GIF")
            else:
                images = load_images("RP.GIF", "RPS.GIF")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.GIF", "BRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("BC.GIF", "BCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("BN.GIF", "BNS.GIF")
            elif isinstance(chess, King):
                images = load_images("BK.GIF", "BKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("BB.GIF", "BBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.GIF", "BAS.GIF")
            else:
                images = load_images("BP.GIF", "BPS.GIF")
        chessman_sprite = Chessman_Sprite(images, chess, w, h)
        sprite_group.add(chessman_sprite)

def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None
