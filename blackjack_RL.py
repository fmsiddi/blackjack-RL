import numpy as np
import random as rnd
import sys

deck = np.tile(np.append(np.arange(1,11), np.tile(10,3)), 4)

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
print('START: sum of {} with initial hand {}'.format(player_sum,player_hand))

# if player_sum > 19:
#     a = 0
# else:
#     a = 1
    
# if a == 1:
    
while player_sum < 20:   
    next_card = rnd.choice(deck)
    if next_card == 1:
        p_ace_in_hand = True
        if player_sum + 11 <= 21:
            next_card = 11
            p_useable_ace = True
    print('HIT:',next_card)
    player_hand = np.append(player_hand,next_card)
    player_sum += next_card
    if player_sum > 21 and (not p_ace_in_hand or (p_ace_in_hand and not p_useable_ace)):
        print('BUST: player sum of {} with deck {}'.format(player_sum,player_hand))
        print('PLAYER LOSES')
        sys.exit()
    elif player_sum > 21 and p_useable_ace:
        player_hand = np.where(player_hand == 11, 1, player_hand) # convert 11 to 1
        p_useable_ace = False
        player_sum = player_hand.sum()
        print('OKAY: player sum of {} with deck {}'.format(player_sum,player_hand))

print('STAND: player sum of {} with deck {}'.format(player_sum,player_hand))

print("\nDealer's turn:")
print('START: sum of {} with initial hand {}'.format(dealer_sum,dealer_hand))
while dealer_sum < 17:
    next_card = rnd.choice(deck)
    if next_card == 1:
        d_ace_in_hand = True
        if dealer_sum + 11 <= 21:
            next_card = 11
            d_useable_ace = True
    print('HIT:',next_card)
    dealer_hand = np.append(dealer_hand,next_card)
    dealer_sum += next_card
    if dealer_sum > 21 and (not d_ace_in_hand or (d_ace_in_hand and not d_useable_ace)):
        print('BUST: dealer sum of {} with deck {}'.format(dealer_sum,dealer_hand))
        print('PLAYER WINS')
        sys.exit()
    elif dealer_sum > 21 and d_useable_ace:
        dealer_hand = np.where(dealer_hand == 11, 1, dealer_hand) # convert 11 to 1
        d_useable_ace = False
        dealer_sum = dealer_hand.sum()
        print('OKAY: dealer sum of {} with deck {}'.format(dealer_sum,dealer_hand))
        
print('STAND: dealer sum of {} with deck {}'.format(dealer_sum,dealer_hand))

if player_sum > dealer_sum:
    print('PLAYER WINS')
elif player_sum < dealer_sum:
    print ('PLAYER LOSES')
else:
    print('TIE')

# first 3 axes: (player sum, dealer sum, useable ace)
# fourth axis = action (0 = stand, 1 = hit)
# π = np.ones(shape = (10,10,2,2)) * 0.5
# Q = np.zeros(shape = (10,10,2,2))

