import random
import itertools as it
import matplotlib.pyplot as plt
import statistics

# print instructions
print(
    "The program uses matplotlib to plot player win percentage (green) vs player average sum at standing (blue). The x-axis is number_of_games/100. Total runtime is around 2 mins, wiith 50000 matches")

# initialize Q-table
bust_limit = 21
dealer_hit_limit = bust_limit - 4

all_sums_player = [i for i in range(0, bust_limit + 10)]
all_sums_dealer = [i for i in range(0, bust_limit + 10)]

player_win_count = 0
dealer_win_count = 0
tie_count = 0
q_table_action_count = 0
random_action_count = 0
total_action_count = 0
to_append = 0
result_log = []

states_list = [p for p in it.product(all_sums_player, repeat=2)]
q_table = dict.fromkeys(states_list, [0, 0])

player_stand = False
player_bust = False
game_over = False
player_sum = 0
state_action = []
learning_rate = 0.05
epsilon = 90
current_state = []
wins_per_n_games = [0]
q_table_action_per_n_games = [0]
random_action_per_n_games = [0]
player_sum_series = [0]
player_sum_series_avg = [0]
incremental_wins_list = []


# change state given action choice
def action(choice):
    global player_sum
    global player_stand
    global player_bust
    global current_state
    global dealer_win_count
    global game_over
    global random_action_count
    global total_action_count
    total_action_count += 1  # twice for random

    if choice == 'hit':  # hit = 1
        state_action_pair = current_state.copy()
        state_action_pair.extend([1])
        state_action.append(state_action_pair)
        new_card = random.randint(1, 10)
        player_sum += new_card

        if player_sum > bust_limit:
            result_log.append(0)
            dealer_win_count += 1
            reward_states(-1)
            player_bust = True
            game_over = True
            player_sum_series.append(player_sum)
    elif choice == 'stand':  # stand = 0
        state_action_pair = current_state.copy()
        state_action_pair.extend([0])
        state_action.append(state_action_pair)
        player_stand = True
        player_sum_series.append(player_sum)
        dealer_play_game()
    elif choice == 'random' and not game_over:
        random_action_count += 1
        if random.randint(1, 2) == 1:
            total_action_count -= 1  # because random has two action() calls
            action('hit')
            return
        else:
            total_action_count -= 1  # because random has two action() calls
            action('stand')
            return


def player_play_game():
    global player_sum
    global dealer_init
    global player_stand
    global current_state
    global game_over
    global q_table_action_count
    global random_action_count
    global player_sum_series

    # draw initial cards
    dealer_init = random.randint(1, 10)
    player_init = random.randint(1, 10)

    current_state = [dealer_init, player_init]
    player_sum = player_init
    while player_sum < bust_limit and not player_stand and not player_bust and not game_over:
        current_state = [dealer_init, player_sum]
        # pick an action from Q-Table with epsilon percent times
        if random.randrange(1, 100) < epsilon:
            # follow Q-table
            q_table_values = q_table[tuple(current_state)]
            if q_table_values[1] > q_table_values[0]:
                # hit
                action('hit')
                q_table_action_count += 1
            elif q_table_values[1] < q_table_values[0]:
                # stand
                action('stand')
                q_table_action_count += 1
            elif q_table_values[1] == q_table_values[0]:
                # random
                action('random')
        else:
            # pick action randomly
            action('random')


def dealer_play_game():
    global player_sum
    global dealer_init
    global player_win_count
    global dealer_win_count
    global game_over
    global tie_count
    # draw initial cards
    dealer_sum = dealer_init
    while dealer_sum < dealer_hit_limit:
        # keep hitting
        dealer_sum += random.randint(1, 10)

    if dealer_sum > bust_limit:
        result_log.append(1)
        player_win_count += 1
        reward_states(1)
        game_over = True
    elif dealer_sum > player_sum:
        result_log.append(0)
        dealer_win_count += 1
        reward_states(-1)
        game_over = True
    elif dealer_sum < player_sum:
        result_log.append(1)
        player_win_count += 1
        reward_states(1)
        game_over = True
    elif dealer_sum == player_sum:
        reward_states(0)
        tie_count += 1
        game_over = True


def reward_states(reward):
    temp = []
    for pair in state_action:
        temp.append(pair)

    for i in range(0, len(temp)):
        state_for_reward = temp[i][:2]
        action_taken = temp[i][-1]

        if i != len(temp) - 1:
            next_state = temp[i + 1][:2]
            next_state_exists = True
        else:
            next_state = [0, 0]
            next_state_exists = False

        # update the q-value
        if next_state_exists:
            new_q_value = q_table[tuple(state_for_reward)][action_taken] + learning_rate * (
                    reward + max(q_table[tuple(next_state)]))
        else:
            new_q_value = q_table[tuple(state_for_reward)][action_taken] + learning_rate * (
                reward)

        if action_taken == 1:
            d = {tuple(state_for_reward): [q_table[tuple(state_for_reward)][0], new_q_value]}
        elif action_taken == 0:
            d = {tuple(state_for_reward): [new_q_value, q_table[tuple(state_for_reward)][1]]}

        q_table.update(d)


for n in range(1, 50000):
    # reset
    game_over = False
    player_stand = False
    player_bust = False
    player_sum = 0
    state_action = []
    current_state = []

    if n % 100 == 0:
        learning_rate = learning_rate * 0.96
        epsilon = epsilon * 1.0005

        # player_sum trend
        last_few_games_sum_avg = statistics.mean(player_sum_series[-1000:])
        player_sum_series_avg.append(last_few_games_sum_avg)
        plt.figure(1)
        plt.plot(player_sum_series_avg, color='blue')
        plt.ylabel('Average player sum at "stand" (blue), Percent wins per last 5000 games (green) ')
        plt.pause(.000005)
        plt.show(block=False)

        # player win ratio trend
        win_percent = statistics.mean(result_log[-5000:]) * 100
        incremental_wins_list.append(win_percent)
        plt.plot(incremental_wins_list, color='green')
        plt.pause(.000005)
        plt.show(block=False)

    player_play_game()

print("Final Q-Table: " + str(q_table))
print("Final player_win_count: " + str(player_win_count) + " : " + str(player_win_count / (
        player_win_count + dealer_win_count + 1) * 100) + "%")
print("Final dealer_win_count: " + str(dealer_win_count) + " : " + str(dealer_win_count / (
        player_win_count + dealer_win_count + 1) * 100) + "%")
