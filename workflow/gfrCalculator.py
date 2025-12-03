import math
import statistics
import numpy as np

class GfrCalculator:

    @staticmethod
    def FAScrea(age: float, sex: str, SCr: float) -> float:
        # Validate input
        if not (2 <= age <= 110):
            raise ValueError("Invalid age")
        if sex not in ("F", "M"):
            raise ValueError("Invalid sex")
        if not (0 < SCr < 30):
            raise ValueError("Invalid SCr")
        try:
            if sex == "M":
                Q = 0.21 + 0.057 * age - 0.0075 * age**2 + 0.00064 * age**3 - 0.000016 * age**4
            else:
                Q = 0.23 + 0.034 * age - 0.0018 * age**2 + 0.00017 * age**3 - 0.0000051 * age**4
            if age > 20:
                Q = 0.9 if sex == "M" else 0.7
            if age < 40:
                result = 107.3 / (SCr / Q)
            else:
                result = 107.3 / (SCr / Q) * (0.988 ** (age - 40))
            return result
        except Exception:
            return None

    @staticmethod
    def FAScreaHt(age: float, sex: str, height: float, SCr: float) -> float:
        if not (2 <= age <= 110):
            return None
        if sex not in ("F", "M"):
            return None
        if not (0 < SCr < 30):
            return None
        if not (50 <= height <= 250):
            return None
        try:
            Q = 3.94 - 13.4 * height / 100 + 17.6 * (height / 100)**2 \
                - 9.84 * (height / 100)**3 + 2.04 * (height / 100)**4
            if age > 20:
                Q = 0.9 if sex == "M" else 0.7
            if age < 40:
                result = 107.3 / (SCr / Q)
            else:
                result = 107.3 / (SCr / Q) * (0.988 ** (age - 40))
            return result
        except Exception:
            return None

    @staticmethod
    def FAScysc(age: float, ScysC: float) -> float:
        if not (2 <= age <= 110):
            return None
        if not (0 < ScysC < 30):
            return None
        try:
            if age < 40:
                result = 107.3 / (ScysC / 0.82)
            elif age < 70:
                result = 107.3 / (ScysC / 0.82) * (0.988 ** (age - 40))
            else:
                result = 107.3 / (ScysC / 0.95) * (0.988 ** (age - 40))
            return result
        except Exception:
            return None

    @staticmethod
    def FAScombi(age: float, sex: str, SCr: float, ScysC: float) -> float:
        if not (2 <= age <= 110):
            return None
        if sex not in ("F", "M"):
            return None
        if not (0 < SCr < 30):
            return None
        if not (0 < ScysC < 30):
            return None
        try:
            if sex == "M":
                Q = 0.21 + 0.057 * age - 0.0075 * age**2 + 0.00064 * age**3 - 0.000016 * age**4
            else:
                Q = 0.23 + 0.034 * age - 0.0018 * age**2 + 0.00017 * age**3 - 0.0000051 * age**4
            if age > 20:
                Q = 0.9 if sex == "M" else 0.7
            if age < 40:
                result = 107.3 / (((SCr / Q) + (ScysC / 0.82)) / 2)
            elif age < 70:
                result = 107.3 / (((SCr / Q) + (ScysC / 0.82)) / 2) * (0.988 ** (age - 40))
            else:
                result = 107.3 / (((SCr / Q) + (ScysC / 0.95)) / 2) * (0.988 ** (age - 40))
            return result
        except Exception:
            return None

    @staticmethod
    def FAScombiHt(age: float, sex: str, height: float, SCr: float, ScysC: float) -> float:
        if not (2 <= age <= 110):
            return None
        if sex not in ("F", "M"):
            return None
        if not (0 < SCr < 30):
            return None
        if not (0 < ScysC < 30):
            return None
        if not (50 <= height <= 250):
            return None
        try:
            Q = 3.94 - 13.4 * height / 100 + 17.6 * (height / 100)**2 \
                - 9.84 * (height / 100)**3 + 2.04 * (height / 100)**4
            if age > 20:
                Q = 0.9 if sex == "M" else 0.7
            if age < 40:
                result = 107.3 / (((SCr / Q) + (ScysC / 0.82)) / 2)
            else:
                result = 107.3 / (((SCr / Q) + (ScysC / 0.82)) / 2) * (0.988 ** (age - 40))
            return result
        except Exception:
            return None

        # @staticmethod
        # def MyPower(Number: float, Exponent: float) -> float:
        #     return Number ** Exponent

    @staticmethod
    def CKDEPIcrea_2009(age: float, sex: str, SCr: float, race: str = "") -> float:
        if not (18 <= age <= 110):
            return None
        if sex not in ("F", "M"):
            return None
        if not (0 < SCr < 30):
            return None
        try:
            if sex == "M":
                if SCr >= 0.9:
                    result = 141 * (SCr / 0.9) ** (-1.209) * (0.9929 ** age)
                else:
                    result = 141 * (SCr / 0.9) ** (-0.411) * (0.9929 ** age)
            else:
                if SCr >= 0.7:
                    result = 144 * (SCr / 0.7) ** (-1.209) * (0.9929 ** age)
                else:
                    result = 144 * (SCr / 0.7) ** (-0.329) * (0.9929 ** age)
            if race == "Black":
                result *= 1.159
            return result
        except Exception:
            return None

    @staticmethod
    def CKDEPI_40(age: float, sex: str, SCr: float) -> float:
        if not (2 <= age <= 110):
            return None
        if sex not in ("F", "M"):
            return None
        if not (0 < SCr < 30):
            return None
        try:
            if age < 40:
                if sex == "M":
                    Q1 = math.log(SCr * 88.4) + 0.259 * (40 - age) - 0.543 * math.log(40 / age) \
                         - 0.00763 * (40**2 - age**2) + 0.000079 * (40**3 - age**3)
                else:
                    Q1 = math.log(SCr * 88.4) + 0.177 * (40 - age) - 0.223 * math.log(40 / age) \
                         - 0.00596 * (40**2 - age**2) + 0.0000686 * (40**3 - age**3)
                Q = math.exp(Q1)
                SCr_adj = Q / 88.4
                age1 = 40
            else:
                SCr_adj = SCr
                age1 = age
            if sex == "M":
                if SCr_adj >= 0.9:
                    result = 141 * (SCr_adj / 0.9) ** (-1.209) * (0.9929 ** age1)
                else:
                    result = 141 * (SCr_adj / 0.9) ** (-0.411) * (0.9929 ** age1)
            else:
                if SCr_adj >= 0.7:
                    result = 144 * (SCr_adj / 0.7) ** (-1.209) * (0.9929 ** age1)
                else:
                    result = 144 * (SCr_adj / 0.7) ** (-0.329) * (0.9929 ** age1)
            return result
        except Exception:
            return None

    @staticmethod
    def CKDEPIcrea_2021(age: float, sex: str, SCr: float) -> float:
        if not (18 <= age <= 110):
            return None
        if sex not in ("F", "M"):
            return None
        if not (0 < SCr < 30):
            return None
        try:
            if sex == "M":
                if SCr >= 0.9:
                    result = 142 * (SCr / 0.9) ** (-1.2) * (0.9938 ** age)
                else:
                    result = 142 * (SCr / 0.9) ** (-0.302) * (0.9938 ** age)
            else:
                if SCr >= 0.7:
                    result = 143 * (SCr / 0.7) ** (-1.2) * (0.9938 ** age)
                else:
                    result = 143 * (SCr / 0.7) ** (-0.241) * (0.9938 ** age)
            return result
        except Exception:
            return None

    @staticmethod
    def LMREV(age: float, sex: str, SCr: float) -> float:
        if age < 18:
            return None
        if sex not in ("F", "M"):
            return None
        if not (SCr > 0):
            return None
        try:
            if sex == "F":
                if SCr < 150 / 88.4:
                    X = 2.5 + 0.0121 * (150 - SCr * 88.4)
                else:
                    X = 2.5 - 0.926 * math.log(SCr * 88.4 / 150)
            else:  # M
                if SCr < 180 / 88.4:
                    X = 2.56 + 0.00968 * (180 - SCr * 88.4)
                else:
                    X = 2.56 - 0.926 * math.log(SCr * 88.4 / 180)
            result = math.exp(X - 0.0158 * age + 0.438 * math.log(age))
            return result
        except Exception: 
            return None

    @staticmethod
    def r_LMR_crea(age: float, sex: str, SCr: float, QM: float, QF: float) -> float:
        if age < 18:
            return None
        if sex not in ("F", "M"):
            return None
        try:
            Ratio = SCr / QM if sex == "M" else SCr / QF
            if Ratio < 2.33:
                X = 4.3087 - 0.7623 * Ratio
            else:
                X = 3.3145 - 0.926 * math.log(Ratio)
            result = math.exp(X - 0.0158 * age + 0.438 * math.log(age))
            return result
        except Exception:
            return None

    @staticmethod
    def r_LMR_cysc(age: float, CysC: float) -> float:
        if age < 2:
            return None
        try:
            Q = 0.83 if age < 50 else 0.83 + 0.005 * (age - 50)
            Ratio = CysC / Q
            if Ratio < 2.33:
                X = 4.3087 - 0.7623 * Ratio
            else:
                X = 3.3145 - 0.926 * math.log(Ratio)
            result = math.exp(X - 0.0158 * age + 0.438 * math.log(age))
            return result
        except Exception:
            return None

    @staticmethod
    def LMR_18(age: float, sex: str, SCr: float) -> float:
        if age < 2:
            return None
        try:
            if age < 18:
                if sex == "M":
                    Q1 = math.log(SCr * 88.4) + 0.259 * (18 - age) - 0.543 * math.log(18 / age) \
                         - 0.00763 * (18**2 - age**2) + 0.000079 * (18**3 - age**3)
                else:
                    Q1 = math.log(SCr * 88.4) + 0.177 * (18 - age) - 0.223 * math.log(18 / age) \
                         - 0.00596 * (18**2 - age**2) + 0.0000686 * (18**3 - age**3)
                Q = math.exp(Q1)
                SCr_adj = Q / 88.4
                age1 = 18
            else:
                SCr_adj = SCr
                age1 = age
            if sex == "F":
                if SCr_adj < 150 / 88.4:
                    X = 2.
        except Exception:
            return None

    @staticmethod
    def EPI_cysC(age: float, sex: str, CysC: float) -> float:
        if age < 2:
            return None
        try:
            if sex=='M':
                if CysC <= 0.8:
                    results= 133 * (CysC / 0.8) ** (-0.499) * (0.996 ** age)
                else:
                    results= 133 * (CysC / 0.8) ** (-1.328) * (0.996 ** age)
            else:
                if CysC <= 0.8:
                    results= 133 * (CysC / 0.8) ** (-0.499) * (0.996 ** age)
                else:
                    results= 144 * (CysC / 0.8) ** (-1.328) * (0.996 ** age)
            return results
        except Exception:
            return None
            
    @staticmethod
    def EPI_Mix_2009(age: float, sex: str, CysC: float, SCr: float) -> float:
        """
        only for patients with age >=18
        """
        if age < 18:
            raise ValueError("Invalid age")
        try:
            if sex=='M':
                if SCr <=0.9:
                    if CysC<=0.8:
                        results= 135 * (SCr / 0.9) ** (-0.207) * ((CysC/0.8) ** (-0.375))*(0.995 ** age)
                    else:
                        results= 135 * (SCr / 0.9) ** (-0.207) * ((CysC/0.8) ** (-0.711))*(0.995 ** age)
                else:
                    if CysC<=0.8:
                        results= 135 * (SCr / 0.9) ** (-0.601) * ((CysC/0.8) ** (-0.375))*(0.995 ** age)
                    else:
                        results= 135 * (SCr / 0.9) ** (-0.601) * ((CysC/0.8) ** (-0.711))*(0.995 ** age)
            else:
                if SCr <=0.7:
                    if CysC<=0.8:
                        results= 130 * (SCr / 0.7) ** (-0.248) * ((CysC/0.8) ** (-0.375))*(0.995 ** age)
                    else:
                        results= 130 * (SCr / 0.7) ** (-0.248) * ((CysC/0.8) ** (-0.711))*(0.995 ** age)
                else:
                    if CysC<=0.8:
                        results= 130 * (SCr / 0.7) ** (-0.601) * ((CysC/0.8) ** (-0.375))*(0.995 ** age)
                    else:
                        results= 130 * (SCr / 0.7) ** (-0.601) * ((CysC/0.8) ** (-0.711))*(0.995 ** age)
            return results
        except Exception:
            return None

    @staticmethod
    def EPI_Mix_2021(age: float, sex: str, SCr: float, CysC: float) -> float:
        """
        Calculates the EPI_Mix_2021 value.
        Returns None if age < 18 or an error occurs.
        """
        if age < 18:
            return None
        try:
            if sex == "M":
                if SCr <= 0.9:
                    if CysC <= 0.8:
                        return 135 * (SCr / 0.9) ** (-0.144) * (CysC / 0.8) ** (-0.323) * (0.9961 ** age)
                    else:  # CysC > 0.8
                        return 135 * (SCr / 0.9) ** (-0.144) * (CysC / 0.8) ** (-0.778) * (0.9961 ** age)
                else:  # SCr > 0.9
                    if CysC <= 0.8:
                        return 135 * (SCr / 0.9) ** (-0.544) * (CysC / 0.8) ** (-0.323) * (0.9961 ** age)
                    else:  # CysC > 0.8
                        return 135 * (SCr / 0.9) ** (-0.544) * (CysC / 0.8) ** (-0.778) * (0.9961 ** age)
            elif sex == "F":
                if SCr <= 0.7:
                    if CysC <= 0.8:
                        return 130 * (SCr / 0.7) ** (-0.219) * (CysC / 0.8) ** (-0.323) * (0.9961 ** age)
                    else:
                        return 130 * (SCr / 0.7) ** (-0.219) * (CysC / 0.8) ** (-0.778) * (0.9961 ** age)
                else:  # SCr > 0.7
                    if CysC <= 0.8:
                        return 130 * (SCr / 0.7) ** (-0.544) * (CysC / 0.8) ** (-0.323) * (0.9961 ** age)
                    else:
                        return 130 * (SCr / 0.7) ** (-0.544) * (CysC / 0.8) ** (-0.778) * (0.9961 ** age)
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def EKFC_Crea(age: float, sex: str, SCr: float) -> float:
        """
        Calculates EKFC_Crea.
        """
        try:
            if sex == "M":
                Q1 = 3.2 + 0.259 * age - 0.543 * math.log(age) - 0.00763 * age**2 + 0.000079 * age**3
            elif sex == "F":
                Q1 = 3.08 + 0.177 * age - 0.223 * math.log(age) - 0.00596 * age**2 + 0.0000686 * age**3
            else:
                return None

            Q = math.exp(Q1) / 88.4

            if age > 25:
                Q = 0.9 if sex == "M" else 0.7

            # Determine exponent based on SCr/Q value and age group
            if SCr /Q < 1:
                if age < 40:
                    return 107.3 / ((SCr / Q) ** 0.322)
                else:
                    return 107.3 / ((SCr / Q) ** 0.322) * (0.99 ** (age - 40))
            else:
                if age < 40:
                    return 107.3 / ((SCr / Q) ** 1.132)
                else:
                    return 107.3 / ((SCr / Q) ** 1.132) * (0.99 ** (age - 40))
        except Exception:
            return None

    @staticmethod
    def EKFC_CysC(age: float, CysC: float) -> float:
        """
        Calculates EKFC_CysC.
        """
        if age < 2:
            return None
        try:
            Q = 0.83
            if age > 50:
                Q = 0.83 + 0.005 * (age - 50)
            if CysC / Q < 1:
                if age < 40:
                    return 107.3 / ((CysC / Q) ** 0.322)
                else:
                    return 107.3 / ((CysC / Q) ** 0.322) * (0.99 ** (age - 40))
            else:
                if age < 40:
                    return 107.3 / ((CysC / Q) ** 1.132)
                else:
                    return 107.3 / ((CysC / Q) ** 1.132) * (0.99 ** (age - 40))
        except Exception:
            return None

    @staticmethod
    def EKFC_Q(age: float, sex: str, SCr: float, QM: float, QF: float) -> float:
        """
        Calculates EKFC_Q.
        Returns None if age < 2 or if QM or QF are out of acceptable ranges.
        """
        if age < 2 or QM < 0.8 or QM > 1.2 or QF < 0.5 or QF > 0.9:
            return None
        try:
            if sex == "M":
                Q1 = 3.2 + 0.259 * age - 0.543 * math.log(age) - 0.00763 * age**2 + 0.000079 * age**3
            elif sex == "F":
                Q1 = 3.08 + 0.177 * age - 0.223 * math.log(age) - 0.00596 * age**2 + 0.0000686 * age**3
            else:
                return None

            Q = math.exp(Q1) / 88.4

            # Calculate adjustment factors
            factor_F = (QF / 0.7 - 1) * (age - 12) / 13 + 1
            factor_M = (QM / 0.9 - 1) * (age - 12) / 13 + 1

            if 12 < age < 25:
                Q = Q * (factor_M if sex == "M" else factor_F)
            if age > 25:
                Q = QM if sex == "M" else QF

            # Determine the result based on SCr/Q and age
            if SCr / Q < 1:
                if age < 40:
                    return 107.3 / ((SCr / Q) ** 0.322)
                else:
                    return 107.3 / ((SCr / Q) ** 0.322) * (0.99 ** (age - 40))
            else:
                if age < 40:
                    return 107.3 / ((SCr / Q) ** 1.132)
                else:
                    return 107.3 / ((SCr / Q) ** 1.132) * (0.99 ** (age - 40))
        except Exception:
            return None

    @staticmethod
    def CAPA(age: float, CysC: float) -> float:
        """
        Calculates CAPA.
        """
        if age < 2:
            return None
        try:
            return 130 * (CysC ** -1.069) * (age ** -0.117) - 7
        except Exception:
            return None

    @staticmethod
    def CG(age: float, sex: str, SCr: float, Wt: float) -> float:
        """
        Calculates the Cockcroft-Gault (CG) value.
        Returns None if age is less than 18.
        """
        if age < 18:
            return None
        try:
            if sex == "M":
                return (140 - age) * Wt / (SCr * 72)
            elif sex == "F":
                return 0.85 * (140 - age) * Wt / (SCr * 72)
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def CKiD(SCr: float, height: float) -> float:
        """
        Calculates CKiD.
        """
        try:
            return 0.413 * height / SCr
        except Exception:
            return None

    @staticmethod
    def CKiDU25Crea(SCr: float, age: float, height: float, sex: str) -> float:
        """
        Calculates CKiDU25Crea.
        Returns None if age is not between 1 and 25.
        """
        if age < 1 or age > 25:
            return None
        try:
            if 1 <= age <= 12:
                if sex == "M":
                    k = 39 * (1.008 ** (age - 12))
                elif sex == "F":
                    k = 36.1 * (1.008 ** (age - 12))
                else:
                    return None
            elif 12 < age <= 18:
                if sex == "M":
                    k = 39 * (1.045 ** (age - 12))
                elif sex == "F":
                    k = 36.1 * (1.023 ** (age - 12))
                else:
                    return None
            elif 18 < age <= 25:
                if sex == "M":
                    k = 50.8
                elif sex == "F":
                    k = 41.4
                else:
                    return None
            else:
                return None
            return k * ((height / 100) / SCr)
        except Exception:
            return None

    @staticmethod
    def CKiDU25CysC(CysC: float, age: float, sex: str) -> float:
        """
        Calculates CKiDU25CysC.
        Returns None if age is not between 1 and 25.
        """
        if age < 1 or age > 25:
            return None
        try:
            if 1 <= age <= 12:
                if sex == "M":
                    k = 87.2 * (1.011 ** (age - 15))
                elif sex == "F":
                    k = 79.9 * (1.004 ** (age - 12))
                else:
                    return None
            elif 12 < age <= 15:
                if sex == "M":
                    k = 87.2 * (1.011 ** (age - 15))
                elif sex == "F":
                    k = 79.9 * (0.974 ** (age - 12))
                else:
                    return None
            elif 15 < age <= 18:
                if sex == "M":
                    k = 87.2 * (0.96 ** (age - 15))
                elif sex == "F":
                    k = 79.9 * (0.974 ** (age - 12))
                else:
                    return None
            elif 18 < age <= 25:
                if sex == "M":
                    k = 77.1
                elif sex == "F":
                    k = 68.3
                else:
                    return None
            else:
                return None
            return k * (1 / CysC)
        except Exception:
            return None
