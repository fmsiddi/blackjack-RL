import numpy as np
import random as rnd
from tqdm import tqdm

def play_hand(policy, hand, ace_in_hand, useable_ace):
    return hand

deck = np.tile(np.append(np.arange(1,11), np.tile(10,3)), 4)

gamma = 1
π = np.ones(shape = (10,10,2,2)) * 0.5
a_space = np.array([0,1])
sa_counter = np.zeros(shape = (10,10,2,2))
sa_returns = np.zeros(shape = (10,10,2,2))
Q = np.zeros(shape = (10,10,2,2))
ε = .1

episodes = 5000000

for e in tqdm(range(episodes)):
    # start of hand
    dealer_hand = np.array(rnd.choice(deck))
    d_ace_in_hand = dealer_hand[dealer_hand == 1].size > 0
    d_useable_ace = d_ace_in_hand
    
    if d_useable_ace:
        dealer_hand = np.where(dealer_hand == 1, 11, dealer_hand) # convert 1 to 11
        
    dealer_sum = dealer_hand.sum()
    
    player_hand = np.array(rnd.choices(deck, k=2))
    p_ace_in_hand = player_hand[player_hand == 1].size > 0
    p_useable_ace = p_ace_in_hand and player_hand.sum() <= 21
    
    if p_useable_ace:
        if player_hand[player_hand == 1].size == 1:
            player_hand = np.where(player_hand == 1, 11, player_hand) # convert 1 to 11
        else:
            rnd_ace_index = rnd.choice(np.argwhere(player_hand == 1))[0]
            player_hand[rnd_ace_index] = 11
        
    player_sum = player_hand.sum()
    while player_sum < 12:
        next_card = rnd.choice(deck)
        if next_card == 1:
            p_ace_in_hand = True
            if player_sum + 11 <= 21:
                next_card = 11
                p_useable_ace = True
        # print('HIT:',next_card)
        player_hand = np.append(player_hand,next_card)
        player_sum += next_card

    # print('\nNEW EPISODE')
    # print('START: sum of {} with initial hand {}'.format(player_sum,player_hand))
    p_bust = False
    d_bust = False
    a = np.random.choice(a_space, p = π[player_sum - 12][dealer_sum - 2][int(p_useable_ace)])
    episode_hist = [(player_sum, dealer_sum, p_useable_ace, a)]
    reward_hist = []
    while a == 1:
        next_card = rnd.choice(deck)
        if next_card == 1:
            p_ace_in_hand = True
            if player_sum + 11 <= 21:
                next_card = 11
                p_useable_ace = True
        # print('HIT:',next_card)
        player_hand = np.append(player_hand, next_card)
        player_sum += next_card    
        if player_sum > 21 and (not p_ace_in_hand or (p_ace_in_hand and not p_useable_ace)):
            # print('BUST: player sum of {} with deck {}'.format(player_sum, player_hand))
            # print('PLAYER LOSES')
            p_bust = True
            R = -1
            reward_hist = np.append(reward_hist, R)
            break
        elif player_sum > 21 and p_useable_ace:
            player_hand = np.where(player_hand == 11, 1, player_hand) # convert 11 to 1
            p_useable_ace = False
            player_sum = player_hand.sum()
            # print('OKAY: player sum of {} with deck {}'.format(player_sum,player_hand))
        R = 0
        reward_hist = np.append(reward_hist, R)
        a = np.random.choice(a_space, p = π[player_sum - 12][dealer_sum - 2][int(p_useable_ace)])
        episode_hist.append((player_sum, dealer_sum, p_useable_ace, a))
    
    if not p_bust:
        # print('STAND: player sum of {} with deck {}'.format(player_sum,player_hand))
        
        # print("\nDealer's turn:")
        # print('START: sum of {} with initial hand {}'.format(dealer_sum,dealer_hand))
        while dealer_sum < 17:
            next_card = rnd.choice(deck)
            if next_card == 1:
                d_ace_in_hand = True
                if dealer_sum + 11 <= 21:
                    next_card = 11
                    d_useable_ace = True
            # print('HIT:',next_card)
            dealer_hand = np.append(dealer_hand, next_card)
            dealer_sum += next_card
            if dealer_sum > 21 and (not d_ace_in_hand or (d_ace_in_hand and not d_useable_ace)):
                # print('BUST: dealer sum of {} with deck {}'.format(dealer_sum,dealer_hand))
                # print('PLAYER WINS')
                R = 1
                reward_hist = np.append(reward_hist, R)
                break
            elif dealer_sum > 21 and d_useable_ace:
                dealer_hand = np.where(dealer_hand == 11, 1, dealer_hand) # convert 11 to 1
                d_useable_ace = False
                dealer_sum = dealer_hand.sum()
                # print('OKAY: dealer sum of {} with deck {}'.format(dealer_sum,dealer_hand))
                
        if not d_bust:
            # print('STAND: dealer sum of {} with deck {}'.format(dealer_sum,dealer_hand))
            if player_sum > dealer_sum:
                # print('PLAYER WINS')
                R = 1
                reward_hist = np.append(reward_hist, R)
            elif player_sum < dealer_sum:
                # print ('PLAYER LOSES')
                R = -1
                reward_hist = np.append(reward_hist, R)
            else:
                # print('TIE')
                R = 0
                reward_hist = np.append(reward_hist, R)
        
    G = 0
    for t in reversed(range(len(episode_hist))):
        G = gamma * G + reward_hist[t]
        p_sum_index = episode_hist[t][0] - 12
        d_sum_index = episode_hist[t][1] - 2
        ua_index = int(episode_hist[t][2])
        a_index = episode_hist[t][3]
        if episode_hist[t] not in episode_hist[:t]:
            sa_returns[p_sum_index][d_sum_index][ua_index][a_index] += G
            sa_counter[p_sum_index][d_sum_index][ua_index][a_index] += 1
            Q[p_sum_index][d_sum_index][ua_index][a_index] = sa_returns[p_sum_index][d_sum_index][ua_index][a_index]/sa_counter[p_sum_index][d_sum_index][ua_index][a_index]
            a_star = np.argmax(Q[p_sum_index][d_sum_index][ua_index])
            π[p_sum_index][d_sum_index][ua_index] = ε/len(a_space)
            π[p_sum_index][d_sum_index][ua_index][a_star] = 1 - ε + ε/len(a_space)
            
while len(tqdm._instances) > 0:
    tqdm._instances.pop().close()
    
ua_policy = np.round(π[:,:,1,1])
nua_policy = np.round(π[:,:,0,1])

# first 3 axes: (player sum, dealer sum, useable ace)
# fourth axis = action (0 = stand, 1 = hit)
# Q = np.zeros(shape = (10,10,2,2))

