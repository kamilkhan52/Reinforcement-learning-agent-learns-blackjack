import random
import itertools as it

# initialize Q-table
all_sums_player = [i for i in range(0, 31)]
all_sums_dealer = [i for i in range(0, 31)]

player_win_count = 0
dealer_win_count = 0

states_list = [p for p in it.product(all_sums_player, repeat=2)]

print("states: " + str(states_list))
q_table = dict.fromkeys(states_list, [0, 0])
print("Q_Table: " + str(q_table))

player_stand = False
player_bust = False
player_sum = 0
state_action = []
learning_rate = 0.10
epsilon = 0.10
current_state = []


# change state given action choice
def action(choice):
    global player_sum
    global player_stand
    global player_bust
    global current_state
    global dealer_win_count

    if choice == 'hit':  # hit = 1
        print("hit")
        state_action_pair = current_state.copy()
        state_action_pair.extend([1])
        state_action.append(state_action_pair)
        print("state_action_pair: " + str(state_action_pair))
        print("state_action: " + str(state_action))
        new_card = random.randint(1, 10)
        print("new_card: " + str(new_card))
        player_sum += new_card
        if player_sum > 21:
            print("player bust! :(. Reward--")
            dealer_win_count += 1
            reward_states(-1)
            player_bust = True
            return
    elif choice == 'stand':  # stand = 0
        print("stand")
        state_action_pair = current_state.copy()
        state_action_pair.extend([0])
        state_action.append(state_action_pair)
        print("state_action_pair: " + str(state_action_pair))
        print("state_action: " + str(state_action))
        player_stand = True
        dealer_play_game()
        return
    elif choice == 'random':
        print("random")
        if random.randint(1, 3) == 1:
            action('hit')
        else:
            action('stand')
            # dealer's turn


def player_play_game():
    global player_sum
    global dealer_init
    global player_stand
    global current_state
    print("Player playing game...")
    # draw initial cards
    dealer_init = random.randint(1, 10)
    player_init = random.randint(1, 10)

    current_state = [dealer_init, player_init]
    print("Current State: " + str(current_state))
    # player to make a choice between hit and stand
    player_sum = player_init
    while player_sum < 21 and not player_stand and not player_bust:
        current_state = [dealer_init, player_sum]
        print("Current State: " + str(current_state))
        # pick an action from Q-Table with 1 - epsilon = 0.90
        if random.randint(1, 10) < 10:
            # follow Q-table
            q_table_values = q_table[tuple(current_state)]
            print("q_table_values: " + str(q_table_values))
            if q_table_values[1] > q_table_values[0]:
                # hit
                action('hit')
                return
            elif q_table_values[1] < q_table_values[0]:
                # stand
                action('stand')
                return
            elif q_table_values[1] == q_table_values[0]:
                # random
                action('random')
                return
        else:
            # pick action randomly
            action('random')
            return


def dealer_play_game():
    global player_sum
    global dealer_init
    global player_win_count
    global dealer_win_count
    print("Dealer playing game...")
    # draw initial cards
    print("Player's sum to beat: " + str(player_sum))
    dealer_sum = dealer_init
    while dealer_sum < 17:
        # keep hitting
        dealer_sum += random.randint(1, 10)

    print("dealer_sum: " + str(dealer_sum))
    if dealer_sum > 21:
        print("dealer bust!")
        player_win_count += 1
        reward_states(1)
    elif dealer_sum > player_sum:
        print("dealer won :( Reward--")
        dealer_win_count += 1
        reward_states(-1)
    elif dealer_sum < player_sum:
        print("player won! :) Reward++")
        player_win_count += 1
        reward_states(1)
    elif dealer_sum == player_sum:
        print("A tie!")


def reward_states(reward):
    next_state_exists = False
    states_for_reward = []
    temp = []
    for pair in state_action:
        temp.append(pair)
    print("temp: " + str(temp))

    for i in range(0, len(temp)):
        state_for_reward = temp[i][:2]
        action_taken = temp[i][-1]

        if i != len(temp) - 1:
            next_state = temp[i + 1][:2]
            next_state_exists = True
            print("next_state: " + str(next_state))
        else:
            next_state = [0, 0]
            next_state_exists = False
            print("No Next state ")

        print("state_for_reward: " + str(state_for_reward))
        print("action_taken: " + str(action_taken))
        print(q_table[tuple(state_for_reward)])
        # update the q-value
        if next_state_exists:
            new_q_value = q_table[tuple(state_for_reward)][action_taken] + learning_rate * (
                    reward + max(q_table[tuple(next_state)]))
        else:
            new_q_value = q_table[tuple(state_for_reward)][action_taken] + learning_rate * (
                reward)

        print("new_q_value: " + str(new_q_value))

        if action_taken == 1:
            d = {tuple(state_for_reward): [q_table[tuple(state_for_reward)][0], new_q_value]}
            print("d: " + str(d))
        elif action_taken == 0:
            d = {tuple(state_for_reward): [new_q_value, q_table[tuple(state_for_reward)][1]]}
            print("d: " + str(d))

        q_table.update(d)

        print("q_table[tuple(state_for_reward)][action_taken]: " + str(q_table[tuple(state_for_reward)][action_taken]))


for n in range(1, 2000000):
    # reset
    player_stand = False
    player_bust = False
    player_sum = 0
    state_action = []
    current_state = []
    player_play_game()

print(q_table)
print("player_win_count: " + str(player_win_count) + " : " + str(player_win_count / (
        player_win_count + dealer_win_count) * 100) + "%")
print("dealer_win_count: " + str(dealer_win_count) + " : " + str(dealer_win_count / (
        player_win_count + dealer_win_count) * 100) + "%")