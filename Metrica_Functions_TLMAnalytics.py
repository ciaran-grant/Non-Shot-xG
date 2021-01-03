# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:55:35 2020

@author: Ciaran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def to_statsbomb_coordinates(data, field_dimen=(120, 80)):
    '''
    Convert positions from Metrica units to StatsBomb (with origin at top left)
    '''
    
    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y']
    data[x_columns] = ( data[x_columns] ) * field_dimen[0]
    data[y_columns] = 80 - ( data[y_columns] ) * field_dimen[1]
    return data


def to_single_playing_direction(home,away,events):
    '''
    Flip StatsBomb coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    for team in [home,away,events]:
        second_half_idx = team.Period.idxmax(2)
        columns_x = [c for c in team.columns if c[-1].lower() in ['x']]
        columns_y = [c for c in team.columns if c[-1].lower() in ['y']]
        team.loc[second_half_idx:,columns_x] = 120 - team.loc[second_half_idx:,columns_x]
        team.loc[second_half_idx:,columns_y] = 80 - team.loc[second_half_idx:,columns_y]

    return home,away,events

def plot_pitch( field_dimen = (120, 80), field_color ='green', linewidth=2, markersize=20, figax = None):
    """ plot_pitch
    
    Plots a soccer pitch. All distance units converted to meters.
    All coordinates and origin converted to StatsBomb.
    
    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (120,80)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    if figax is None: # create new pitch 
        fig,ax = plt.subplots(figsize=(12,8)) # create a figure 
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color=='green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke' # line color
        pc = 'w' # 'spot' colors
    elif field_color=='white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3,3) # include a border arround of the field of width 3m
    meters_per_yard = 0.9144 # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2. # length of half pitch
    half_pitch_width = field_dimen[1]/2. # width of half pitch
    signs = [-1,1] 
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard
    # plot half way line # center circle
    ax.plot([60,60],[0,80],lc,linewidth=linewidth)
    ax.scatter(60,40,marker='o',facecolor=lc,linewidth=0,s=markersize)
    y = np.linspace(-1,1,50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x+60,y+40,lc,linewidth=linewidth)
    ax.plot(-x+60,y+40,lc,linewidth=linewidth)
    for s in signs: # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length+60,half_pitch_length+60],[s*half_pitch_width+40,s*half_pitch_width+40],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length+60,s*half_pitch_length+60],[-half_pitch_width+40,half_pitch_width+40],lc,linewidth=linewidth)
        # goal posts & line
        ax.plot( [s*half_pitch_length+60,s*half_pitch_length+60],[40-goal_line_width/2.,40+goal_line_width/2.],pc+'s',markersize=6*markersize/20.,linewidth=linewidth)
        # 6 yard box
        ax.plot([s*half_pitch_length+60,s*half_pitch_length-s*box_length+60],[40+box_width/2.,40+box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length+60,s*half_pitch_length-s*box_length+60],[40-box_width/2.,40-box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*box_length+60,s*half_pitch_length-s*box_length+60],[40-box_width/2.,40+box_width/2.],lc,linewidth=linewidth)
        # penalty area
        ax.plot([s*half_pitch_length+60,s*half_pitch_length-s*area_length+60],[40+area_width/2.,40+area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length+60,s*half_pitch_length-s*area_length+60],[40-area_width/2.,40-area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*area_length+60,s*half_pitch_length-s*area_length+60],[40-area_width/2.,40+area_width/2.],lc,linewidth=linewidth)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot+60,40,marker='o',facecolor=lc,linewidth=0,s=markersize)
        # corner flags
        y = np.linspace(0,1,50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x+60,-half_pitch_width+y+40,lc,linewidth=linewidth)
        ax.plot(s*half_pitch_length-s*x+60,half_pitch_width-y+40,lc,linewidth=linewidth)
        # draw the D
        y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x+60,y+40,lc,linewidth=linewidth)
        
    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0] + 60
    ymax = field_dimen[1]/2. + border_dimen[1] + 40
    ax.set_xlim([0-border_dimen[0],xmax])
    ax.set_ylim([0-border_dimen[1],ymax])
    # Setting origin to top left as per StatsBomb
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    return fig,ax

def plot_frame( hometeam, awayteam, figax=None, team_colors=('r','b'), field_dimen = (120, 80), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False ):
    """ plot_frame( hometeam, awayteam )
    
    Plots a frame of Metrica tracking data (player positions and the ball) on a football pitch. All distances should be in meters.
    Field dimensions default to StatsBomb 120x80
    
    Parameters
    -----------
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    # plot home & away teams in order
    for team,color in zip( [hometeam,awayteam], team_colors) :
        x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
        y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
        ax.plot( team[x_columns], team[y_columns], color+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
        if annotate:
            [ ax.text( team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color  ) for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 
    # plot ball
    ax.plot( hometeam['ball_x'], hometeam['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
    return fig,ax

def plot_events( events, figax=None, field_dimen = (120,80), indicators = ['Marker','Arrow'], color='r', marker_style = 'o', alpha = 0.5, annotate=False):
    """ plot_events( events )
    
    Plots Metrica event positions on a football pitch. event data can be a single or several rows of a data frame. All distances should be in meters (or according to StatsBomb's coordinates).
    Field dimensions defaulted to StatsBomb 120x80.
    
    Parameters
    -----------
        events: row (i.e. instant) of the home team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        indicators: List containing choices on how to plot the event. 'Marker' places a marker at the 'Start X/Y' location of the event; 'Arrow' draws an arrow from the start to end locations. Can choose one or both.
        color: color of indicator. Default is 'r' (red)
        marker_style: Marker type used to indicate the event position. Default is 'o' (filled ircle).
        alpha: alpha of event marker. Default is 0.5    
        annotate: Boolean determining whether text annotation from event data 'Type' and 'From' fields is shown on plot. Default is False.
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """

    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax 
    for i,row in events.iterrows():
        if 'Marker' in indicators:
            ax.plot(  row['Start X'], row['Start Y'], color+marker_style, alpha=alpha )
        if 'Arrow' in indicators:
            ax.annotate("", xy=row[['End X','End Y']], xytext=row[['Start X','Start Y']], alpha=alpha, arrowprops=dict(alpha=alpha,width=0.5,headlength=4.0,headwidth=4.0,color=color),annotation_clip=False)
        if annotate:
            textstring = row['Type'] + ': ' + row['From']
            ax.text( row['Start X'], row['Start Y'], textstring, fontsize=10, color=color)
    return fig,ax

# Calculate xG for each frame
def calculate_xG(shot):
    ''' calculate_xG (shot)
    
    Calculates the Expected Goals based on a model trained using StatsBomb event and freeze frame data.
    Input is a row of a dataframe with columns for distance, angle, distance_nearest_defender, number_blocking_defenders
    
    '''
    # For the model 'b', get the intercept
    intercept=1.0519
    # For as many variables as put in the model, 
    # bsum = intercept + (coefficient * variable value)
    bsum=intercept + 0.1080*shot['distance'] - 1.6109*shot['angle'] - 0.1242*shot['distance_nearest_defender'] + 0.3260*shot['number_blocking_defenders']
    # Calculate probability of goal as 1 / 1 + exp(model output)
    xG = 1/(1+np.exp(bsum)) 
    return xG

# Decide if defender is between shooting location and goal
def area(x1, y1, x2, y2, x3, y3): 
    '''
    Calculates the area inside a triangle with coordinates (x1, y1), (x2, y2), (x3, y3)
    '''
  
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)  
                + x3 * (y1 - y2)) / 2.0) 

def is_inside(x1, y1, x2, y2, x3, y3, x, y): 
    '''
    Checks if (x, y) is inside triangle (x1, y1), (x2, y2), (x3, y3).
    If area of triangle == sum of areas of each point with (x, y).
    Then point is inside.
    
    Returns boolean.
    
    Example:
    is_inside(0, 0, 20, 0, 10, 30, 10, 15)

    '''
    # Calculate area of triangle ABC 
    A = area (x1, y1, x2, y2, x3, y3) 
  
    # Calculate area of triangle PBC  
    A1 = area (x, y, x2, y2, x3, y3) 
      
    # Calculate area of triangle PAC  
    A2 = area (x1, y1, x, y, x3, y3) 
      
    # Calculate area of triangle PAB  
    A3 = area (x1, y1, x2, y2, x, y) 
      
    # Check if sum of A1, A2 and A3  
    # is same as A 
    if(A == A1 + A2 + A3): 
        return True
    else: 
        return False
    
def create_features(tracking_home, tracking_away,
                    start_frame, number_frames,
                    field_dimen = (120, 80),
                    attacking_team = 'Home',
                    playing_direction = 'right-left'
                    ):
    
    '''
    Creates features required for calculate_xG from StatsBomb coordinate tracking data.
    Returns single tracking dataframe with home and away locations + features.
    '''
    # Filter only necessary frames and merge to get single dataframe (want to access attacking and defending locations)
    tracking_home_frames = tracking_home.iloc[start_frame:start_frame+number_frames]
    tracking_away_frames = tracking_away.iloc[start_frame:start_frame+number_frames]
    tracking_frames = pd.merge(tracking_home_frames, tracking_away_frames, how='left', on = ['Period', 'Time [s]', 'ball_x', 'ball_y'])

    # Determine which way shooting towards, Home = right -> left, Away = left -> right
    if attacking_team == 'Home':
        defending_team = 'Away'
    elif attacking_team == 'Away':
        defending_team = 'Home'    
        
    if playing_direction == 'right-left':
        left_post_x = 0
        left_post_y = 40+(8*0.9144)/2
        right_post_x = 0
        right_post_y = 40-(8*0.9144)/2
        centre_goal_x = 0
        centre_goal_y = 40
    elif playing_direction == 'left-right':        
        left_post_x = 120
        left_post_y = 40-(8*0.9144)/2
        right_post_x = 120
        right_post_y = 40+(8*0.9144)/2
        centre_goal_x = 120
        centre_goal_y = 40
    
    misc_columns = ['Period', 'Time [s]', 'ball_x', 'ball_y']
    player_cols = set(tracking_frames.columns) - set(misc_columns)
    defender_cols = list(set([int(player.split('_')[1]) for player in player_cols if attacking_team not in player]))

    for index, row in tracking_frames.iterrows():

        # Ball locations
        ball_x = tracking_frames.loc[index, 'ball_x']
        ball_y = tracking_frames.loc[index, 'ball_y']

        # Need to calculate the distance from ball to centre of goal
        tracking_frames.loc[index, 'distance'] = np.sqrt((ball_x - centre_goal_x)**2 + (ball_y - centre_goal_y)**2)

        # Calculate the angle:
        a = np.sqrt((ball_x - right_post_x)**2 + (ball_y - right_post_y)**2)
        b = np.sqrt((left_post_x - right_post_x)**2 + (left_post_y - right_post_y)**2)
        c = np.sqrt((left_post_x - ball_x)**2 + (left_post_y - ball_y)**2)
        angle_ac = np.arccos((a**2 + c**2 - b**2)/(2*a*c))
        if angle_ac<0:
            angle_ac=np.pi+angle_ac
        tracking_frames.loc[index,'angle'] =angle_ac

        # Defender info
        distance_nearest_defender = None
        blocking_defenders = []
        for defender in defender_cols:
            column_x = '{}_{}_x'.format(defending_team, defender)
            column_y = '{}_{}_y'.format(defending_team, defender)
            defender_x = tracking_frames.loc[index, column_x]
            defender_y = tracking_frames.loc[index, column_y]

            # Distance to nearest defender
            distance = np.sqrt((ball_x - defender_x)**2 + (ball_y - defender_y)**2)
            if distance_nearest_defender == None:
                distance_nearest_defender = distance
            elif distance < distance_nearest_defender:
                distance_nearest_defender = distance

            # Is the defender between ball and the goal (inside the triangle between ball, left, right post)?
            blocking = is_inside(ball_x, ball_y
                                , left_post_x, left_post_y
                                , right_post_x, right_post_y
                                , defender_x, defender_y)
            # If defender is blocking part of the goal, add to list
            if blocking:
                blocking_defenders.append(defender)

        tracking_frames.loc[index, 'distance_nearest_defender'] = distance_nearest_defender
        tracking_frames.loc[index, 'number_blocking_defenders'] = len(blocking_defenders)
    
    return tracking_frames

def create_xG(tracking_frames):
    '''
    Creates xG column by applying calculate_xG to each row
    
    '''
    
    for index, row in tracking_frames.iterrows():
        xG = calculate_xG(row)
        tracking_frames.loc[index, 'xG'] = xG
    
    return tracking_frames

def create_non_shot_xG(tracking_frames, 
                       attacking_team = 'Home',
                       distance_threshold = 1):
    
    '''
    Creates non_shot_xG column in tracking_frames.
    This column is equal to xG when an attacking player is within distance_threshold of the ball (eg. they are able to shoot)
    And missing (np.nan) otherwise.
    This will ensure gaps in plots when xG isn't applicable.
    '''
    
    # Define attacking or defending teams to get correct set of attacking players (numbers and columns)
    misc_columns = ['Period', 'Time [s]', 'ball_x', 'ball_y']
    player_cols = set(tracking_frames.columns) - set(misc_columns)
    attacking_players = list(set([int(player.split('_')[1]) for player in player_cols if attacking_team in player]))
    
    for index, row in tracking_frames.iterrows():
        ball_x = tracking_frames.loc[index, 'ball_x']
        ball_y = tracking_frames.loc[index, 'ball_y']
        #print('Ball: {}, {}'.format(ball_x, ball_y))

        dist_ball_attacker = None
        for attacker in attacking_players:
            
            column_x = '{}_{}_x'.format(attacking_team, attacker)
            column_y = '{}_{}_y'.format(attacking_team, attacker)
            attacker_x = tracking_frames.loc[index, column_x]
            attacker_y = tracking_frames.loc[index, column_y]
            
            # Calculate distance to the ball
            distance = np.sqrt((ball_x - attacker_x)**2 + (ball_y - attacker_y)**2)
            
            # If first defender or distance is less than current lowest, then overwrite
            if dist_ball_attacker is None:
                dist_ball_attacker = distance
            elif distance < dist_ball_attacker:
                dist_ball_attacker = distance

        # Update tracking frame with distance to nearest attacker
        tracking_frames.loc[index, 'distance_nearest_attacker'] = dist_ball_attacker

        # If the ball is within distance_threshold of attacker then assume they could shoot and so assign xG, otherwise make missing
        if (dist_ball_attacker > 0) & (dist_ball_attacker < distance_threshold):
            tracking_frames.loc[index, 'non_shot_xG'] = tracking_frames.loc[index, 'xG']
        else:
            tracking_frames.loc[index, 'non_shot_xG'] = np.nan
    
    return tracking_frames

def save_match_clip_xg(tracking_frames, fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('r','b'), field_dimen = (120, 80), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7):
    """ save_match_clip( hometeam, awayteam, fpath )
    
    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'.
    
    Updated to work for StatsBomb coordinates and single tracking_frames dataframe as input.
    Multiplte subplots are created, one on top for the pitch and below with the xG time series.
    
    Parameters
    -----------
        tracking_frames: tracking data DataFrame. Movie will be created from all rows in the DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    # From single tracking_frames dataframe, get hometeam and awayteam
    misc_columns = ['Period', 'Time [s]', 'ball_x', 'ball_y']
    player_cols = set(tracking_frames.columns) - set(misc_columns)

    home_players = list(set([player.split('_')[1] for player in player_cols if 'Home' in player]))
    home_x = ['Home_{}_x'.format(player) for player in home_players]
    home_y = ['Home_{}_y'.format(player) for player in home_players]
    home_cols = misc_columns[:2] + home_x + home_y + misc_columns[2:]
    hometeam = tracking_frames[home_cols]

    away_players = list(set([player.split('_')[1] for player in player_cols if 'Away' in player]))
    away_x = ['Away_{}_x'.format(player) for player in away_players]
    away_y = ['Away_{}_y'.format(player) for player in away_players]
    away_cols = misc_columns[:2] + away_x + away_y + misc_columns[2:]
    awayteam = tracking_frames[away_cols]

    # check that indices match first
    assert np.all( hometeam.index==awayteam.index ), "Home and away team Dataframe indices must be the same"
    # in which case use home team index
    index = hometeam.index
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' +  fname + '.mp4' # path and filename
    
    # create figure with pitch 4/5 rows and xG plot 1/5 rows at bottom
    if figax is None:
        fig = plt.figure(figsize = (12, 10))
        gs = fig.add_gridspec(nrows=5, ncols=1)

        ax1 = fig.add_subplot(gs[0:4])
        fig,ax1 = plot_pitch( field_dimen = field_dimen, figax = (fig, ax1))

        ax2 = fig.add_subplot(gs[4])
        x_data = []
        y_data1 = []
        y_data2 = []
        line1, = ax2.plot(x_data, y_data1, lw=2, color='r')
        line2, = ax2.plot(x_data, y_data2, lw=2, color='green')

        # limits are the time (x-axis)
        ax2.set_xlim(tracking_frames['Time [s]'].min(), tracking_frames['Time [s]'].max())
        # limits are xG %
        ax2.set_ylim(0, 1)
        
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
     
    
    # Generate movie
    print("Generating movie...",end='')
    with writer.saving(fig, fname, 100):
        for i in index:
            # Pitch Movie
            figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team,color in zip( [hometeam.loc[i],awayteam.loc[i]], team_colors) :
                x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
                y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
                objs, = ax1.plot( team[x_columns], team[y_columns], color+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                    objs = ax1.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                    figobjs.append(objs)
            # plot ball
            objs, = ax1.plot( team['ball_x'], team['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
            figobjs.append(objs)
            # include match time at the top
            frame_minute =  int( team['Time [s]']/60. )
            frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
            timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
            objs = ax1.text(-5+60,field_dimen[1]/2.+1.-40, timestring, fontsize=14 )
            figobjs.append(objs)

            # xG Plot
            x = tracking_frames.loc[i, 'Time [s]']
            x_data.append(x)
            y1 = tracking_frames.loc[i, 'xG']
            y_data1.append(y1)
            line1.set_data(x_data, y_data1)
            y2 = tracking_frames.loc[i, 'xG_available']
            y_data2.append(y2)
            line2.set_data(x_data, y_data2)
        
            writer.grab_frame()
            
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()
            
    print("done")
    plt.clf()
    plt.close(fig) 