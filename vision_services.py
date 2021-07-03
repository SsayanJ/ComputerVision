from Aruco_signs import return_opponent_positions, setup_for_match, girouette, proj_pos, record_match

from cellaserv.service import Service
from cellaserv.proxy import CellaservProxy

class Vision(Service):

    def __init__(self):
        super().__init__()
        self.ortho_proj = None
        self.opponent_ids = None
    #     #self.cs = CellaservProxy()
    #     #self.match_color = self.cs.match.get_color()
        self.proj_positions = None
        self.girouette_c = None

    @Service.action("setup")
    def setup_match_vision(self, team_color='yellow'):
        self.ortho_proj, self.opponent_ids = setup_for_match(team_color)
        return "vision setup: ok"

    @Service.action
    def get_opp(self):
        return self.opponent_ids.__str__()


    @Service.action
    def girouette_color(self, img, team_color):
        self.girouette_c = girouette(img, team_color)
        return self.girouette_c

    @Service.action
    def get_opponents_position(self, img, balise_height):
        _, positions = return_opponent_positions(
            img, self.ortho_proj, self.opponent_ids)
        proj_positions = proj_pos(positions, balise_height)
        self.proj_positions = proj_positions
        return str(proj_positions)

    # output file path to be checked

    @Service.action
    def record_match_serv(self, match_duration=100):
        record_match(match_duration)


def main():
    vision_service = Vision()
    vision_service.run()


if __name__ == "__main__":
    main()
