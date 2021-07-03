from Aruco_signs import return_opponent_positions, setup_for_match, girouette, proj_pos, record_match

from cellaserv.services import Service


@Service
def setup_match_vision(team_color):
    return setup_for_match(team_color)


@Service
def girouette_color(img, team_color):
    return girouette(img, "yellow")


@Service
def get_opponents_position(img, ortho_proj, opponent_ids, balise_height):
    _, positions = return_opponent_positions(
        img, ortho_proj, opponent_ids)
    proj_positions = proj_pos(positions, balise_height)
    return proj_positions

# output file path to be checked


@Service
def record_match_serv(match_duration=100):
    record_match(match_duration)
