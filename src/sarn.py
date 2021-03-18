"""
Arbitrary datasets for our semantics project (SARN)
"""

import argparse, time
from infer import SentenceBase, Knowledge
from getMono import CCGtrees, ErrorCCGtree, ErrorCompareSemCat
from sklearn.metrics import confusion_matrix
import spacy
from pass2act import P2A_transformer
import mywordnet

# setup logger: https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
import logging

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


def setup_logger(name, log_file, level=logging.INFO, formatter=formatter):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(
        log_file + "_" + time.strftime("%Y_%m_%d-%H_%M_%S"), mode="w+"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def main():
    # -------------------------------------
    # parse cmd arguments
    description = """
    Solve sick. Author: Hai Hu, huhai@indiana.edu
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-r",
        dest="n_rep",
        type=int,
        default=2,
        help="number of replacements [default: %(default)s]",
    )
    parser.add_argument(
        "-k",
        dest="print_k",
        action="store_const",
        const=True,
        default=False,
        help="if -k, print k [default: %(default)s]",
    )
    parser.add_argument(
        "-g",
        dest="gen_inf",
        action="store_const",
        const=True,
        default=False,
        help="if -g, generate infs, neutrals and contras, do not solve the problems [default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        dest="pred_log",
        action="store_const",
        const=True,
        default=False,
        help="if -l, then save predictions in a log file: pred_monalog_DATE_TIME [default: %(default)s]",
    )
    parser.add_argument(
        "-b",
        dest="backward",
        action="store_const",
        const=True,
        default=False,
        help="if -b, then reverse P and H = backward [default: %(default)s]",
    )

    args = parser.parse_args()
    # -------------------------------------

    solver = Solver(
        args.n_rep,
        args.print_k,
        args.gen_inf,
        args.pred_log,
        args.backward,
    )

    solver.solve()


class Solver:
    def __init__(self, n_rep, print_k, gen_inf, pred_log, backward):
        self.n_rep = n_rep
        self.print_k = print_k
        self.p2a = P2A_transformer(spacy.load("en_core_web_sm"))
        self.gen_inf = gen_inf
        self.nlp = spacy.load("en_core_web_lg")  # make sure to use larger model!
        self.pred_log = pred_log
        if self.pred_log:
            self.pred_logger = setup_logger(
                "pred_logger", "pred_monalog.txt", formatter=""
            )
        self.reverse_P_H = backward  # do backward inference

    def P_idx(self, id):
        """ return idx of P """
        if self.reverse_P_H:
            return id * 2 + 1
        return id * 2

    def H_idx(self, id):
        """ return idx of H in sick2uniq """
        if self.reverse_P_H:
            return id * 2
        return id * 2 + 1

    def solve(self):
        start_time = time.time()

        trees = CCGtrees(fn_log="med_adjectives.tok.preprocess.log")

        # read parsed trees from different parsers
        trees.readEasyccgStr("med_adjectives.easyccg.parsed.txt")
        # trees.readEasyccgStr("sick_uniq.raw.depccg.parsed.txt")

        ids = list(range(0, 10))

        self.solve_helper(ids, trees)

        print("\n\n--- %s seconds ---" % (time.time() - start_time))

    def solve_helper(self, ids, trees):
        """ ids: a list of ids to solve """
        y_pred = []

        for id_to_solve in ids:
            ans = self.solve_one(id_to_solve, trees, reverse=False)
            # if the ans is "U", try going from hypothesis to premise, see if get "C"
            if not self.gen_inf:  # if not only generate infs and contras
                if ans == "U":
                    ans_rev = self.solve_one(id_to_solve, trees, reverse=True)
                    if ans_rev in [
                        "C",
                        "E_pass",
                    ]:  # reverse will only count when E_pass
                        ans = ans_rev
                if ans == "E_pass":
                    ans = "E"  # E_pass = passive to active
            y_pred.append(ans)
            if self.pred_log:
                self.pred_logger.info("{}\t{}".format(id_to_solve, ans))

        print("\ny_pred:", y_pred)

    def solve_one(self, id_to_solve, trees, reverse=False):
        """solve problem
        steps:
            1. read in parsed Ps and H
            2. initialize knowledge base K, update K when reading in Ps and H
            3. do 3 replacement for each P, store all unique inferences in INF
            4. if H in INF: entail, else if: ... contradict, else: unknown
        """
        print("-" * 30)
        print("\n*** solving sick {} ***\n".format(id_to_solve))

        # -----------------------------
        # readin Ps and H
        # build the tree here
        use_lemma = True
        if self.gen_inf:
            use_lemma = False
        if not reverse:
            P = trees.build_one_tree(self.P_idx(id_to_solve), "easyccg", use_lemma)
            H = trees.build_one_tree(self.H_idx(id_to_solve), "easyccg", use_lemma)
        else:
            P = trees.build_one_tree(self.H_idx(id_to_solve), "easyccg", use_lemma)
            H = trees.build_one_tree(self.P_idx(id_to_solve), "easyccg", use_lemma)

        # -----------------------------
        # initialize s
        s = SentenceBase(gen_inf=self.gen_inf)

        # -----------------------------
        # build knowledge
        k = Knowledge()
        k.build_quantifier()  # all = every = each < some = a = an, etc.
        # k.build_morph_tense()         # man = men, etc.

        # fix trees and update knowledge k, sentBase s
        # P
        P.fixQuantifier()
        P.fixNot()
        # if parser == 'candc': t.fixRC()  # only fix RC for candc
        k.update_sent_pattern(P)  # patterns like: every X is NP
        k.update_word_lists(P)  # nouns, subSecAdj, etc.
        s.add_P_str(P)

        # H
        H.fixQuantifier()
        H.fixNot()
        k.update_word_lists(H)  # need to find nouns, subSecAdjs, etc. in H
        s.add_H_str_there_be(H)  # transform ``there be'' in H

        # -----------------------------
        k.build_manual_for_sick()
        mywordnet.assign_all_relations_wordnet(k)
        # -----------------------------

        k.update_modifier()  # adj + n < n, n + RC/PP < n, v + PP < v
        # k.update_rules()            # if N < VP and N < N_1, then N < N_1 who VP,

        s.k = k

        if self.print_k:
            k.print_knowledge()

        # exit()
        # -----------------------------
        # polarize
        print("\n*** polarizing ***\n")
        for p in s.Ps_ccgtree:  # Ps
            try:
                p.mark()
                p.polarize()
                p.getImpSign()
            except (ErrorCCGtree, ErrorCompareSemCat) as e:  # , AssertionError
                print("cannot polarize:", p.wholeStr)
                print("error:", e)
                return type(e).__name__
            except AssertionError as e:
                print("assertion error!")
                print(e)
                p.printSent()
                return type(e).__name__
                # exit()
            # print("P: ", end="")
            # p.printSent_raw_no_pol()
            print("P: ", end="")
            p.printSent()
        print("H: ", end="")
        s.H_ccgtree.printSent_raw_no_pol()
        # s.H_ccgtree.printSent()

        # -----------------------------
        # replacement
        # iterative deepening search (IDS)
        print("\n*** replacement ***\n")
        ans = s.solve_ids(depth_max=self.n_rep, fracas_id=None)

        print("\n--- tried ** {} ** inferences:".format(len(s.inferences)))
        # for inf in sorted(s.inferences): print(inf)
        for inf in s.inferences_tree:
            inf.printSent_raw()

        print("\n--- tried ** {} ** contradictions:".format(len(s.contras_str)))
        for contra in sorted(s.contras_str):
            print(contra.lower())

        print("\n*** H: ", end="")
        print(s.H)

        # make decision
        print("\n*** decision ***")
        print("y_pred:", ans)

        return ans


if __name__ == "__main__":
    main()
