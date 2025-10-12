# dam_selfplay_two_nets_english_rightlog.py
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque

# -----------------------
# Config / hyperparams
# -----------------------
BOARD_SIZE = 8
SQUARE_SIZE = 80
LOG_WIDTH = 300
FPS = 20
DEVICE = torch.device("cpu")

EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
REPLAY_CAPACITY = 5000
BATCH_SIZE = 64
TRAIN_EPOCHS_PER_GAME = 3
LEARNING_RATE = 1e-3
TOTAL_GAMES = 100

# GUI sizes
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
BOARD_HEIGHT = BOARD_SIZE * SQUARE_SIZE
WIDTH = BOARD_WIDTH + LOG_WIDTH
HEIGHT = BOARD_HEIGHT
LOG_LINE_HEIGHT = 18
MAX_LOG_LINES = HEIGHT // LOG_LINE_HEIGHT

# -----------------------
# Pygame init
# -----------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers AI Self-Play")
font = pygame.font.SysFont("arial", 16)
bigfont = pygame.font.SysFont("arial", 22)
clock = pygame.time.Clock()

# -----------------------
# Board representation
# -----------------------
def start_board():
    b = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    for y in range(3):
        for x in range(BOARD_SIZE):
            if (x+y)%2==1:
                b[y][x] = -1
    for y in range(5,8):
        for x in range(BOARD_SIZE):
            if (x+y)%2==1:
                b[y][x] = 1
    return b

def clone_board(board):
    return [row[:] for row in board]

def board_to_tensor(board, perspective=1):
    flat = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            flat.append(board[y][x]*perspective)
    return torch.tensor(flat, dtype=torch.float32, device=DEVICE)

# -----------------------
# Move generation
# -----------------------
def in_bounds(x,y): return 0<=x<BOARD_SIZE and 0<=y<BOARD_SIZE
def piece_color(p): return 1 if p>0 else -1 if p<0 else 0
def is_king(p): return abs(p)==2
def directions_for(p):
    if is_king(p): return [(-1,-1),(1,-1),(-1,1),(1,1)]
    return [(-1,-1),(1,-1)] if p>0 else [(-1,1),(1,1)]

def find_simple_moves(board,x,y):
    p = board[y][x]
    moves=[]
    for dx,dy in directions_for(p):
        nx,ny = x+dx,y+dy
        if in_bounds(nx,ny) and board[ny][nx]==0:
            moves.append(((x,y),(nx,ny)))
    return moves

def find_captures_from(board,x,y):
    results=[]
    p = board[y][x]
    color = piece_color(p)
    def rec(cur_board,cx,cy,path,captured_any):
        found=False
        for dx,dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx,ny = cx+dx,cy+dy
            jx,jy = cx+2*dx,cy+2*dy
            if in_bounds(nx,ny) and in_bounds(jx,jy):
                target = cur_board[ny][nx]
                landing = cur_board[jy][jx]
                if target!=0 and piece_color(target)==-color and landing==0:
                    newb = clone_board(cur_board)
                    newb[jy][jx]=newb[cy][cx]
                    newb[cy][cx]=0
                    newb[ny][nx]=0
                    rec(newb,jx,jy,path+[((cx,cy),(jx,jy),(nx,ny))],True)
                    found=True
        if not found and captured_any:
            results.append(path)
    rec(board,x,y,[],False)
    return results

def get_all_moves(board,player):
    captures, simples=[], []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if piece_color(board[y][x])==player:
                caps = find_captures_from(board,x,y)
                for seq in caps:
                    seq_pairs = [(m[0],m[1]) for m in seq]
                    captures.append((True, seq_pairs))
                if not caps:
                    for m in find_simple_moves(board,x,y):
                        simples.append([ (m[0],m[1]) ])
    return captures if captures else [(False, s) for s in simples]

def apply_move(board, move_seq):
    b = clone_board(board)
    for (x1,y1),(x2,y2) in move_seq:
        piece = b[y1][x1]
        if abs(x2-x1)==2 and abs(y2-y1)==2:
            mx,my = (x1+x2)//2,(y1+y2)//2
            b[my][mx]=0
        b[y2][x2] = piece
        b[y1][x1]=0
    for x in range(BOARD_SIZE):
        if b[0][x]==1: b[0][x]=2
        if b[BOARD_SIZE-1][x]==-1: b[BOARD_SIZE-1][x]=-2
    return b

def game_result(board):
    w = sum(1 for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x]>0)
    b = sum(1 for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board[y][x]<0)
    if w==0: return -1
    if b==0: return 1
    return None

# -----------------------
# Neural Networks
# -----------------------
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(BOARD_SIZE*BOARD_SIZE,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
        )
    def forward(self,x): return self.net(x)

# -----------------------
# Agent / policy
# -----------------------
def select_move_by_nn(board, player, net, epsilon):
    moves = get_all_moves(board, player)
    if not moves: return None
    if random.random()<epsilon:
        choice=random.choice(moves)
        seq = choice[1]
        if isinstance(seq[0][0],int): seq=[seq]
        return (seq, choice[0])
    best_val=-float('inf')
    best=None
    for is_cap, seq in moves:
        if isinstance(seq[0][0],int): seq=[seq]
        newb = apply_move(board,seq)
        t = board_to_tensor(newb,perspective=player)
        with torch.no_grad():
            val=net(t.unsqueeze(0)).item()
        if val>best_val:
            best_val=val
            best=(seq,is_cap)
    return best

# -----------------------
# Replay Buffer
# -----------------------
class ReplayBuffer:
    def __init__(self,capacity):
        self.buf=deque(maxlen=capacity)
    def push(self,state,outcome):
        self.buf.append((state.cpu(),outcome))
    def sample(self,batch_size):
        import random
        batch=random.sample(self.buf,min(len(self.buf),batch_size))
        states=torch.stack([item[0] for item in batch]).to(DEVICE)
        targets=torch.tensor([item[1] for item in batch],dtype=torch.float32,device=DEVICE).unsqueeze(1)
        return states,targets
    def __len__(self): return len(self.buf)

# -----------------------
# Drawing
# -----------------------
def draw_board(board):
    screen.fill((50,50,50))
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            rect = pygame.Rect(x*SQUARE_SIZE,y*SQUARE_SIZE,SQUARE_SIZE,SQUARE_SIZE)
            color = (232,220,200) if (x+y)%2==0 else (125,82,55)
            pygame.draw.rect(screen,color,rect)
            p=board[y][x]
            if p!=0:
                cx,cy=rect.center
                radius=24 if abs(p)==1 else 28
                pygame.draw.circle(screen,(245,245,245) if p>0 else (30,30,30),(cx,cy),radius)
                if abs(p)==2:
                    lbl = bigfont.render("K",True,(200,10,10) if p>0 else (200,200,200))
                    screen.blit(lbl,(cx-10,cy-18))

def draw_log(log_lines, scroll_offset=0):
    x0 = BOARD_WIDTH + 10
    y0 = 10
    # bereken zichtbare lijnen
    start_idx = max(0,len(log_lines)-MAX_LOG_LINES-scroll_offset)
    end_idx = len(log_lines)-scroll_offset
    visible_lines = log_lines[start_idx:end_idx]
    # achtergrond
    pygame.draw.rect(screen,(30,30,30),(BOARD_WIDTH,0,LOG_WIDTH,HEIGHT))
    for i,ln in enumerate(visible_lines):
        screen.blit(font.render(ln,True,(200,200,200)),(x0,y0+i*LOG_LINE_HEIGHT))

# -----------------------
# Main setup
# -----------------------
net_white = ValueNet().to(DEVICE)
net_black = ValueNet().to(DEVICE)
optimizer_white = optim.Adam(net_white.parameters(),lr=BATCH_SIZE)
optimizer_black = optim.Adam(net_black.parameters(),lr=BATCH_SIZE)
replay_white = ReplayBuffer(REPLAY_CAPACITY)
replay_black = ReplayBuffer(REPLAY_CAPACITY)

epsilon = EPSILON_START
board = start_board()
turn = 1
move_count=0
wins_white=0
wins_black=0
draws=0
loss_white=0
loss_black=0
log_lines=[]
game_states_white=[]
game_states_black=[]
games_played = 0
scroll_offset=0

def append_log(s):
    log_lines.append(s)
    print(s)

def finish_game(result):
    global wins_white,wins_black,draws,loss_white,loss_black,epsilon,games_played
    if result==1: wins_white+=1
    elif result==-1: wins_black+=1
    else: draws+=1
    for st in game_states_white:
        outcome = 1.0 if result==1 else -1.0 if result==-1 else 0.0
        replay_white.push(st,outcome)
    for st in game_states_black:
        outcome = 1.0 if result==-1 else -1.0 if result==1 else 0.0
        replay_black.push(st,outcome)
    # train
    if len(replay_white)>=min(64,BATCH_SIZE):
        for _ in range(TRAIN_EPOCHS_PER_GAME):
            states,targets = replay_white.sample(BATCH_SIZE)
            preds=net_white(states)
            loss=nn.MSELoss()(preds,targets)
            optimizer_white.zero_grad()
            loss.backward()
            optimizer_white.step()
    if len(replay_black)>=min(64,BATCH_SIZE):
        for _ in range(TRAIN_EPOCHS_PER_GAME):
            states,targets = replay_black.sample(BATCH_SIZE)
            preds=net_black(states)
            loss=nn.MSELoss()(preds,targets)
            optimizer_black.zero_grad()
            loss.backward()
            optimizer_black.step()
    epsilon=max(EPSILON_END,epsilon*EPSILON_DECAY)
    game_states_white.clear()
    game_states_black.clear()
    games_played+=1

append_log("Start self-play between WhiteNet and BlackNet.")

# -----------------------
# Main loop
# -----------------------
running=True
while running and games_played<TOTAL_GAMES:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
        elif event.type==pygame.MOUSEBUTTONDOWN:
            if event.button==4:  # scroll up
                scroll_offset = min(scroll_offset+1,len(log_lines)-MAX_LOG_LINES)
            elif event.button==5: # scroll down
                scroll_offset = max(scroll_offset-1,0)

    moves = get_all_moves(board, turn)
    if not moves:
        result = -turn
        append_log(f"No moves for {'White' if turn==1 else 'Black'} -> winner: {'White' if result==1 else 'Black'}")
        finish_game(result)
        board = start_board()
        turn=1
        move_count=0
        continue

    chosen = select_move_by_nn(board, turn, net_white if turn==1 else net_black, epsilon)
    if chosen is None:
        result=-turn
        append_log(f"No move found for {'White' if turn==1 else 'Black'}")
        finish_game(result)
        board=start_board()
        turn=1
        move_count=0
        continue

    seq,is_cap = chosen
    if isinstance(seq[0][0],int): seq=[seq]

    st = board_to_tensor(board, perspective=turn)
    if turn==1: game_states_white.append(st.detach().cpu())
    else: game_states_black.append(st.detach().cpu())

    board = apply_move(board,seq)
    move_count+=1

    res = game_result(board)
    if res is not None:
        append_log(f"Game finished after {move_count} moves. Winner: {'White' if res==1 else 'Black'}")
        finish_game(res)
        board = start_board()
        turn=1
        move_count=0
        continue

    if move_count>=200:
        append_log(f"Draw (move limit) after {move_count} moves")
        finish_game(0)
        board=start_board()
        turn=1
        move_count=0
        continue

    turn*=-1

    draw_board(board)
    draw_log(log_lines, scroll_offset)

    if wins_white>wins_black:
        pygame.display.set_caption(f"White AI leading {wins_white}-{wins_black}")
    elif wins_black>wins_white:
        pygame.display.set_caption(f"Black AI leading {wins_black}-{wins_white}")
    else:
        pygame.display.set_caption(f"Tied: {wins_white}-{wins_black}")

    pygame.display.flip()
    clock.tick(FPS)

append_log("Reached total games limit. Closing...")
time.sleep(2)
pygame.quit()
