"""
=============================================================================
  Staff Optimization & Burnout Prevention  Enhanced Data Generator v3
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
import time

# ---------------------------------------------------------------------------
# VERSION 3 CHANGELOG (LOGIC IMPROVEMENTS  NO SCHEMA CHANGE)
# ---------------------------------------------------------------------------
# 1. Status-aware timesheet generation
#    - Shifts are generated only for staff with status = 'Active'.
#    - Prevents inactive employees from appearing in workforce analytics.

# 2. Unified date window for all operational tables
#    - Timesheet, patient_assignment, and leave_records now use the same
#      configured date window: TIMESHEET_START_DATE  TIMESHEET_END_DATE.
#    - Ensures consistent time-based analysis across datasets.

# 3. Department-specific patient case distributions
#    - Patient case types are now sampled using department-level
#      probability distributions instead of a global distribution.
#    - Example:
#        ER  more Emergency cases
#        ICU  higher critical workload
#        General Ward  mostly Routine cases
#    - Produces realistic workload patterns.

# 4. Leave eligibility constraints
#    - Leave records are generated only for staff who are not 'Inactive'.
#    - Prevents unrealistic leave records for terminated employees.

# 5. Leave vs shift conflict prevention
#    - Leave generation checks timesheet records and adjusts leave dates
#      to avoid overlap with worked shifts.
#    - Ensures staff cannot appear both working and on leave simultaneously.

# 6. Cross-department staffing simulation
#    - Introduced department cross-posting using DEPT_CROSS_POST_RATE.
#    - Simulates real hospital scenarios where staff temporarily work
#      outside their home department during demand spikes.

# 7. Multi-patient assignments per shift
#    - Each shift can generate multiple patient assignments
#      (13 patients per shift with configurable probabilities).
#    - Produces more realistic workload distribution.

# 8. Multi-year surge modeling
#    - Surge windows now generated for multiple years (20242026).
#    - Enables seasonal and year-over-year burnout analysis.

# 9. Driver-based burnout escalation model
#    - Replaced random high-risk cohort with driver-based burnout boost.
#    - Burnout susceptibility now depends on:
#          staff role
#          department risk level
#          experience level
#          overtime load
#          night shifts
#          surge periods
#    - Creates more realistic burnout distributions.

# 10. Improved department alignment between shifts and patient assignments
#     - Patient assignments generated using the shift department context.
#     - Ensures department consistency across operational tables.

# ---------------------------------------------------------------------------

OUTPUT_DIR          = "generated_data"
STAFF_COUNT         = 500_000        # staff_master rows
TIMESHEET_COUNT     = 3_000_000      # staff_timesheet rows
OTHER_COUNT         = 200_000        # transactional table rows (richer variety)
DEPT_CROSS_POST_RATE = 0.15          # % of shifts moved from home dept
ASSIGNMENTS_PER_SHIFT_OPTIONS = [1, 2, 3]
ASSIGNMENTS_PER_SHIFT_PROBS   = [0.65, 0.25, 0.10]
TIMESHEET_START_DATE = date(2024, 1, 1)
TIMESHEET_END_DATE   = date(2026, 2, 28)
SEED                = 42
np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def _t(): return time.time()


#  Surge windows (emergency periods that cause burnout spikes) 
SURGE_YEARS = [2024, 2025, 2026]
SURGE_WINDOW_TEMPLATES = [
    ((1, 15),  (1, 25)),   # Flu surge Jan
    ((3, 1),   (3, 10)),   # Trauma spike Mar
    ((5, 5),   (5, 15)),   # Seasonal spike May
    ((7, 20),  (7, 30)),   # Heatwave Jul
    ((9, 10),  (9, 20)),   # Post-holiday surge Sep
]


def build_surge_windows(years):
    windows = []
    for y in years:
        for (sm, sd), (em, ed) in SURGE_WINDOW_TEMPLATES:
            windows.append((date(y, sm, sd), date(y, em, ed)))
    return windows


SURGE_WINDOWS = build_surge_windows(SURGE_YEARS)

def is_surge(d: date) -> bool:
    for s, e in SURGE_WINDOWS:
        if s <= d <= e:
            return True
    return False


#  Salary-band helper 
# Band is determined by per_hour_rate  40 hrs/week  52 weeks (annual estimate)
def salary_band_from_rate(rates: np.ndarray) -> np.ndarray:
    annual = rates * 40 * 52
    return np.select(
        [annual < 400_000, annual < 700_000, annual < 1_100_000, annual < 1_600_000],
        ['Band A',         'Band B',          'Band C',           'Band D'],
        default='Band E'
    )


# 1. staff_master
def generate_staff_master():
    t0 = _t()
    n  = STAFF_COUNT
    print(f"Generating staff_master ({n:,} rows)")

    first_names_m = ['Amit', 'Raj', 'Vikram', 'Suresh', 'Manoj', 'Ravi', 'Nikhil', 'Arun',
                     'John', 'Michael', 'David', 'Robert', 'James', 'Daniel', 'Samuel', 'Kevin']
    first_names_f = ['Priya', 'Anita', 'Sunita', 'Rekha', 'Neha', 'Pooja', 'Divya', 'Lakshmi',
                     'Emily', 'Sarah', 'Linda', 'Karen', 'Jessica', 'Nancy', 'Lisa', 'Susan']
    last_names    = ['Sharma', 'Patel', 'Singh', 'Kumar', 'Verma', 'Gupta', 'Rao', 'Nair',
                     'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']

    roles         = ['Nurse', 'Doctor', 'Admin', 'Surgeon', 'Radiologist',
                     'Pharmacist', 'Physiotherapist', 'Technician']
    # Role  experience level mapping probability (realistic)
    exp_levels    = ['Junior', 'Senior', 'Experienced']

    departments   = ['ICU', 'ER', 'General Ward', 'Pediatrics',
                     'Oncology', 'Cardiology', 'Radiology', 'Pathology']
    shift_prefs   = ['Day', 'Night', 'Any']
    statuses      = ['Active', 'Inactive', 'On Leave']
    locations     = ['Central Hospital', 'North Clinic', 'East Annex', 'West Medical Center']

    # Gender assignment ( 55% Female in healthcare)
    gender        = np.random.choice(['Female', 'Male'], n, p=[0.55, 0.45])

    # Build names respecting gender
    fn = np.where(
        gender == 'Female',
        np.random.choice(first_names_f, n),
        np.random.choice(first_names_m, n)
    )
    ln = np.random.choice(last_names, n)
    staff_name = np.char.add(np.char.add(fn, ' '), ln)

    role          = np.random.choice(roles, n)

    #  Experience level: vectorized using role-to-prob mapping 
    # Map each role to an integer index for fast lookup
    exp_probs_list = {
        'Doctor':          [0.20, 0.45, 0.35],
        'Surgeon':         [0.05, 0.35, 0.60],
        'Nurse':           [0.30, 0.45, 0.25],
        'Technician':      [0.40, 0.40, 0.20],
        'Admin':           [0.40, 0.40, 0.20],
        'Radiologist':     [0.20, 0.40, 0.40],
        'Pharmacist':      [0.30, 0.40, 0.30],
        'Physiotherapist': [0.35, 0.40, 0.25],
    }
    # Assign experience per role group (vectorized per unique role)
    experience_level = np.empty(n, dtype=object)
    for r, probs in exp_probs_list.items():
        mask = (role == r)
        cnt  = mask.sum()
        if cnt > 0:
            experience_level[mask] = np.random.choice(exp_levels, cnt, p=probs)

    #  Per-hour rates: vectorized 
    base_rate_map  = {
        'Doctor': 800, 'Surgeon': 1200, 'Radiologist': 900, 'Pharmacist': 600,
        'Nurse': 350,  'Physiotherapist': 400, 'Technician': 300, 'Admin': 250
    }
    exp_multiplier = {'Junior': 1.0, 'Senior': 1.4, 'Experienced': 1.8}
    role_base  = np.array([base_rate_map[r]        for r in role])
    exp_mult   = np.array([exp_multiplier.get(e, 1.0) for e in experience_level])
    per_hour_rate = (role_base * exp_mult * np.random.uniform(0.90, 1.10, n)).round(2)

    #  Salary band: vectorized cutpoints 
    salary_band = salary_band_from_rate(per_hour_rate)

    #  hire_date: vectorize with date arithmetic 
    hire_date_offsets = np.random.randint(0, 3650, n)
    base_hire = datetime(2015, 1, 1).date()
    hire_dates = [base_hire + timedelta(days=int(d)) for d in hire_date_offsets]

    shift_preference = np.random.choice(shift_prefs, n, p=[0.50, 0.30, 0.20])
    emp_base         = np.random.choice(['Full-time', 'Contract', 'Part-time'], n, p=[0.65, 0.25, 0.10])
    employment_type  = np.char.add(np.char.add(emp_base, '/'), gender)

    df = pd.DataFrame({
        'staff_id':           [f"STF_{i:07d}" for i in range(1, n + 1)],
        'staff_name':         staff_name,
        'gender':             gender,
        'role':               role,
        'experience_level':   experience_level,
        'department':         np.random.choice(departments, n),
        'hire_date':          hire_dates,
        'employment_type':    employment_type,
        'shift_preference':   shift_preference,
        'status':             np.random.choice(statuses, n, p=[0.90, 0.05, 0.05]),
        'base_location':      np.random.choice(locations, n),
        'per_hour_rate':      per_hour_rate,
        'salary_band':        salary_band,
    })

    # night_shift_female_flag will be back-filled after timesheet is generated
    df['night_shift_female_flag'] = 0   # placeholder; updated later

    path = os.path.join(OUTPUT_DIR, "staff_master.parquet")
    df.to_parquet(path, index=False)
    print(f"staff_master -> {path} ({time.time()-t0:.1f}s)")
    return df


# 
# 2. staff_timesheet
# 
def generate_staff_timesheet(staff_df: pd.DataFrame, pa_df: pd.DataFrame = None):
    t0  = _t()
    n   = TIMESHEET_COUNT
    print(f"Generating staff_timesheet ({n:,} rows)")

    departments = ['ICU', 'ER', 'General Ward', 'Pediatrics',
                   'Oncology', 'Cardiology', 'Radiology', 'Pathology']

    staff_ids    = staff_df['staff_id'].values
    gender_map   = dict(zip(staff_df['staff_id'], staff_df['gender']))
    staff_dept_map = dict(zip(staff_df['staff_id'], staff_df['department']))
    staff_status_map = dict(zip(staff_df['staff_id'], staff_df['status']))
    # Per-hour rate lookup for overtime pay calculation
    rate_map     = dict(zip(staff_df['staff_id'], staff_df['per_hour_rate']))

    active_staff_ids = np.array([sid for sid in staff_ids if staff_status_map.get(sid) == 'Active'])
    if len(active_staff_ids) == 0:
        raise ValueError("No Active staff available for timesheet generation")

    selected_ids = np.random.choice(active_staff_ids, n, replace=True)
    home_departments = np.array([staff_dept_map[sid] for sid in selected_ids], dtype=object)
    cross_post_flag = np.random.choice([0, 1], n, p=[1 - DEPT_CROSS_POST_RATE, DEPT_CROSS_POST_RATE])
    random_departments = np.random.choice(departments, n)
    department_override = np.where(cross_post_flag == 1, random_departments, home_departments)

    #  Shift dates spanning configured window (through Feb 2026) 
    base_date = datetime.combine(TIMESHEET_START_DATE, datetime.min.time())
    window_days = (TIMESHEET_END_DATE - TIMESHEET_START_DATE).days + 1
    date_offsets = [int(d) for d in np.random.randint(0, window_days, n)]
    shift_dates = [base_date + timedelta(days=d) for d in date_offsets]

    # If patient assignments are not provided, generate them linked to these shifts
    # (same staff_id + same shift_date) without changing schema.
    auto_generated_pa = pa_df is None
    if auto_generated_pa:
        pa_df = generate_patient_assignment(
            staff_ids=staff_ids,
            linked_staff_ids=selected_ids,
            linked_dates=[d.date() for d in shift_dates],
            linked_departments=department_override.tolist()
        )

    #  Shift start: 07:00 (day) or 19:00 (night) 
    shift_type_arr = np.random.choice(['Day', 'Night'], n, p=[0.60, 0.40])

    starts = [d + timedelta(hours=(7 if st == 'Day' else 19))
              for d, st in zip(shift_dates, shift_type_arr)]

    #  Shift duration: 812 hrs, with 12-hr bursts during surges 
    duration_hrs = []
    for d, st in zip(shift_dates, shift_type_arr):
        if is_surge(d.date()):
            dur = int(np.random.choice([10, 11, 12, 12, 12], p=[0.1, 0.2, 0.25, 0.25, 0.2]))
        else:
            dur = int(np.random.randint(8, 13))
        duration_hrs.append(dur)

    ends = [s + timedelta(hours=h) for s, h in zip(starts, duration_hrs)]

    #  Overtime hours = max(0, shift_duration - 8) 
    overtime_hrs = np.array([max(0.0, h - 8.0) for h in duration_hrs]).round(2)

    #  Break minutes 
    break_min = np.random.randint(20, 61, n)

    #  Surge flags & Night flags (needed for burnout formula) 
    surge_flags = np.array([1 if is_surge(d.date()) else 0 for d in shift_dates])
    night_flags = (shift_type_arr == 'Night').astype(int)

    # 
    # BURNOUT SCORE FORMULA
    # 
    #
    #   burnout_score = (overtime_hours   2)
    #                 + (critical_cases   3)     from patient_assignment
    #                 + (night_shift      5)     1 if Night, 0 if Day
    #                 + (consecutive_days 4)     1 if >4 consecutive days
    #                 + (emergency_cases  3)     from patient_assignment
    #
    #   Capped at 100.  Risk labels:
    #     030   Low  |  3155  Medium  |  5675  High  |  76100  Critical
    # 

    # Component 1: overtime_hours  2
    c_overtime = (overtime_hrs * 2)

    # Component 2 & 5: critical_cases  3  and  emergency_cases  3
    # Look up from patient_assignment (per staff per shift_date)
    if pa_df is not None:
        # Count critical cases per staff per date
        crit_agg = (
            pa_df[pa_df['critical_flag'] == 1]
            .groupby(['staff_id', 'assignment_date'])
            .size().reset_index(name='critical_cases')
        )
        # Count emergency cases per staff per date
        emrg_agg = (
            pa_df[pa_df['case_type'] == 'Emergency']
            .groupby(['staff_id', 'assignment_date'])
            .size().reset_index(name='emergency_cases')
        )
        # Build lookup series keyed by (staff_id, date)
        crit_map = dict(zip(
            zip(crit_agg['staff_id'], crit_agg['assignment_date']),
            crit_agg['critical_cases']
        ))
        emrg_map = dict(zip(
            zip(emrg_agg['staff_id'], emrg_agg['assignment_date']),
            emrg_agg['emergency_cases']
        ))
        shift_dates_d = [d.date() for d in shift_dates]
        critical_cases_arr  = np.array([crit_map.get((sid, dt), 0)
                                        for sid, dt in zip(selected_ids, shift_dates_d)])
        emergency_cases_arr = np.array([emrg_map.get((sid, dt), 0)
                                        for sid, dt in zip(selected_ids, shift_dates_d)])
    else:
        # Fallback: simulate with random values (03)
        critical_cases_arr  = np.random.randint(0, 4, n)
        emergency_cases_arr = np.random.randint(0, 4, n)

    c_critical  = critical_cases_arr  * 3
    c_emergency = emergency_cases_arr * 3

    # Component 3: night_shift  5
    c_night = night_flags * 5

    # Component 4: consecutive_days  4
    # True consecutive-day detection (>4 calendar days in a row).
    df_tmp = pd.DataFrame({
        'idx': np.arange(n),
        'staff_id': selected_ids,
        'shift_date': pd.to_datetime([d.date() for d in shift_dates]),
    })

    # Deduplicate same-day multi-shifts before streak detection.
    unique_days = (
        df_tmp[['staff_id', 'shift_date']]
        .drop_duplicates()
        .sort_values(['staff_id', 'shift_date'])
    )
    unique_days['prev_date'] = unique_days.groupby('staff_id')['shift_date'].shift(1)
    unique_days['is_new_streak'] = (
        (unique_days['shift_date'] - unique_days['prev_date']).dt.days.ne(1)
    ).fillna(True)
    unique_days['streak_id'] = unique_days.groupby('staff_id')['is_new_streak'].cumsum()
    unique_days['streak_len'] = unique_days.groupby(
        ['staff_id', 'streak_id']
    )['shift_date'].transform('size')
    unique_days['consecutive_days_flag'] = (unique_days['streak_len'] > 4).astype(int)

    # Map date-level streak flags back to original timesheet rows.
    flag_map = unique_days.set_index(['staff_id', 'shift_date'])['consecutive_days_flag']
    consecutive_days_flag = (
        pd.Series(
            df_tmp.set_index(['staff_id', 'shift_date']).index.map(flag_map),
            index=df_tmp.index
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    c_consecutive = consecutive_days_flag * 4

    # Final burnout score  capped at 100
    burnout_raw   = c_overtime + c_critical + c_night + c_consecutive + c_emergency
    burnout_score = np.clip(burnout_raw, 0, 100).astype(int)

    # Risk label
    burnout_risk = np.select(
        [burnout_score <= 30, burnout_score <= 55, burnout_score <= 75],
        ['Low',               'Medium',             'High'],
        default='Critical'
    )

    #  Driver-based chronic burnout lift (replaces flat random 5% cohort) 
    role_weight = {
        'Nurse': 1.0, 'Doctor': 0.8, 'Surgeon': 1.1, 'Radiologist': 0.7,
        'Pharmacist': 0.6, 'Physiotherapist': 0.6, 'Technician': 0.7, 'Admin': 0.4
    }
    dept_weight = {
        'ICU': 1.1, 'ER': 1.1, 'Oncology': 0.9, 'Cardiology': 0.8,
        'General Ward': 0.6, 'Pediatrics': 0.6, 'Radiology': 0.4, 'Pathology': 0.4
    }
    exp_weight = {'Junior': 0.9, 'Senior': 0.7, 'Experienced': 0.6}

    staff_role_map = dict(zip(staff_df['staff_id'], staff_df['role']))
    staff_exp_map  = dict(zip(staff_df['staff_id'], staff_df['experience_level']))

    role_arr = np.array([staff_role_map[s] for s in selected_ids], dtype=object)
    exp_arr  = np.array([staff_exp_map[s]  for s in selected_ids], dtype=object)

    susceptibility = (
        np.vectorize(role_weight.get)(role_arr, 0.6) * 0.4 +
        np.vectorize(dept_weight.get)(home_departments, 0.5) * 0.4 +
        np.vectorize(exp_weight.get)(exp_arr, 0.7) * 0.2
    )
    acute_load = (
        0.20 * overtime_hrs +
        0.70 * night_flags +
        0.80 * surge_flags +
        0.90 * consecutive_days_flag
    )
    chronic_prob = np.clip(0.01 + 0.06 * susceptibility + 0.03 * acute_load, 0.0, 0.45)
    chronic_flag = (np.random.rand(n) < chronic_prob).astype(int)

    # Moderate additive uplift; preserves observed workload signal while adding persistence.
    chronic_boost = chronic_flag * np.random.randint(8, 23, n)
    burnout_score = np.clip(burnout_score + chronic_boost, 0, 100).astype(int)
    burnout_risk = np.select(
        [burnout_score <= 30, burnout_score <= 55, burnout_score <= 75],
        ['Low',               'Medium',             'High'],
        default='Critical'
    )

    #  Back-to-back 12-hour shift flag 
    # Flag only when:
    # 1) current shift is >=12h
    # 2) previous shift for same staff is >=12h
    # 3) gap between previous end and current start is <=24 hours
    b2b_tmp = pd.DataFrame({
        'idx': np.arange(n),
        'staff_id': selected_ids,
        'shift_start': pd.to_datetime(starts),
        'shift_end': pd.to_datetime(ends),
        'shift_duration_hours': np.array(duration_hrs, dtype=float),
    }).sort_values(['staff_id', 'shift_start', 'shift_end'])

    b2b_tmp['prev_shift_end'] = b2b_tmp.groupby('staff_id')['shift_end'].shift(1)
    b2b_tmp['prev_shift_duration'] = b2b_tmp.groupby('staff_id')['shift_duration_hours'].shift(1)
    b2b_tmp['gap_hours'] = (
        (b2b_tmp['shift_start'] - b2b_tmp['prev_shift_end']).dt.total_seconds() / 3600.0
    )

    b2b_tmp['back_to_back_12h'] = (
        (b2b_tmp['shift_duration_hours'] >= 12) &
        (b2b_tmp['prev_shift_duration'] >= 12) &
        (b2b_tmp['gap_hours'] >= 0) &
        (b2b_tmp['gap_hours'] <= 24)
    ).astype(int)

    back_to_back_flag = (
        b2b_tmp.set_index('idx')['back_to_back_12h']
        .reindex(np.arange(n))
        .fillna(0)
        .astype(int)
        .values
    )

    #  Night-to-day flip flag 
    # Sort by staff + date; if prev row same staff is Night and current is Day  1
    ts_df_temp = pd.DataFrame({
        'idx':        np.arange(n),
        'staff_id':   selected_ids,
        'shift_date': [d.date() for d in shift_dates],
        'shift_type': shift_type_arr
    }).sort_values(['staff_id', 'shift_date'])
    ts_df_temp['prev_shift']     = ts_df_temp.groupby('staff_id')['shift_type'].shift(1)
    ts_df_temp['night_to_day']   = (
        (ts_df_temp['prev_shift'] == 'Night') & (ts_df_temp['shift_type'] == 'Day')
    ).astype(int)
    # Map back to original index
    night_day_flip = ts_df_temp.set_index('idx')['night_to_day'].reindex(np.arange(n)).fillna(0).astype(int).values

    #  Female night flag 
    female_night_flag = np.array([
        1 if (gender_map.get(sid, '') == 'Female' and st == 'Night') else 0
        for sid, st in zip(selected_ids, shift_type_arr)
    ])

    # 
    # OVERTIME PAY
    # 
    #
    #   overtime_pay = overtime_hours  per_hour_rate  overtime_multiplier
    #
    #   Multiplier tiers:
    #     0   hrs overtime    0    (no extra pay)
    #     02 hrs overtime    1.5 (Time and a Half)
    #     24 hrs overtime    2.0 (Double Pay)
    #     >4  hrs overtime    2.5 (Emergency / Critical Rate)
    #
    # 
    ot_multiplier = np.select(
        [overtime_hrs == 0,
         overtime_hrs <= 2,
         overtime_hrs <= 4],
        [0.0, 1.5, 2.0],
        default=2.5
    )
    per_hour_arr = np.array([rate_map.get(sid, 300.0) for sid in selected_ids])
    overtime_pay = (overtime_hrs * per_hour_arr * ot_multiplier).round(2)

    df = pd.DataFrame({
        'timesheet_id':           [f"TS_{i:08d}" for i in range(1, n + 1)],
        'staff_id':               selected_ids,
        'shift_date':             [d.date() for d in shift_dates],
        'shift_start':            starts,
        'shift_end':              ends,
        'shift_type':             shift_type_arr,
        'shift_duration_hours':   [round(h, 2) for h in duration_hrs],
        'overtime_hours':         overtime_hrs,
        'break_minutes':          break_min,
        'department_override':    department_override,
        'logged_hours_flag':      np.random.choice([True, False], n, p=[0.95, 0.05]),
        'burnout_score':          burnout_score,
        'burnout_risk_label':     burnout_risk,
        'surge_period_flag':      surge_flags,
        'consecutive_days_flag':  consecutive_days_flag,
        'back_to_back_12h_flag':  back_to_back_flag,
        'night_to_day_flip_flag': night_day_flip,
        'female_night_flag':      female_night_flag,
        'overtime_multiplier':    ot_multiplier,
        'overtime_pay':           overtime_pay,
    })

    path = os.path.join(OUTPUT_DIR, "staff_timesheet.parquet")
    df.to_parquet(path, index=False)
    if auto_generated_pa:
        home_vs_shift_mismatch = (department_override != home_departments).mean() * 100
        print(f"Dept QA: home->shift mismatch={home_vs_shift_mismatch:.2f}% | shift->assignment mismatch=0.00% (linked)")
    print(f"staff_timesheet -> {path} ({time.time()-t0:.1f}s)")
    return df


# 
# 3. patient_assignment
# 
def generate_patient_assignment(
    staff_ids: np.ndarray,
    linked_staff_ids: np.ndarray = None,
    linked_dates: list = None,
    linked_departments: list = None
):
    t0  = _t()
    n   = OTHER_COUNT

    departments = ['ICU', 'ER', 'General Ward', 'Pediatrics', 'Oncology', 'Cardiology']
    case_types  = ['Emergency', 'Routine', 'Surgery', 'Consultation', 'Follow-up']
    dept_case_probs = {
        'ICU':          [0.45, 0.20, 0.20, 0.10, 0.05],
        'ER':           [0.55, 0.20, 0.10, 0.10, 0.05],
        'General Ward': [0.10, 0.55, 0.10, 0.15, 0.10],
        'Pediatrics':   [0.18, 0.45, 0.07, 0.20, 0.10],
        'Oncology':     [0.12, 0.35, 0.10, 0.18, 0.25],
        'Cardiology':   [0.25, 0.35, 0.18, 0.12, 0.10],
        'Radiology':    [0.05, 0.30, 0.10, 0.40, 0.15],
        'Pathology':    [0.03, 0.32, 0.05, 0.45, 0.15],
    }
    default_case_probs = [0.20, 0.40, 0.12, 0.18, 0.10]

    if linked_staff_ids is not None and linked_dates is not None:
        if len(linked_staff_ids) != len(linked_dates):
            raise ValueError("linked_staff_ids and linked_dates must have the same length")
        base_selected = np.array(linked_staff_ids)
        base_dates = np.array([datetime.combine(d, datetime.min.time()) for d in linked_dates], dtype=object)
        base_n = len(base_selected)

        assignments_per_shift = np.random.choice(
            ASSIGNMENTS_PER_SHIFT_OPTIONS, base_n, p=ASSIGNMENTS_PER_SHIFT_PROBS
        ).astype(int)

        selected = np.repeat(base_selected, assignments_per_shift)
        dates = np.repeat(base_dates, assignments_per_shift)
        n = len(selected)

        if linked_departments is not None:
            if len(linked_departments) != base_n:
                raise ValueError("linked_departments must match linked_staff_ids length")
            base_departments = np.array(linked_departments, dtype=object)
            assignment_departments = np.repeat(base_departments, assignments_per_shift)
        else:
            assignment_departments = np.random.choice(departments, n)
    else:
        selected   = np.random.choice(staff_ids, n)
        base_date = datetime.combine(TIMESHEET_START_DATE, datetime.min.time())
        window_days = (TIMESHEET_END_DATE - TIMESHEET_START_DATE).days + 1
        dates = [base_date + timedelta(days=int(d)) for d in np.random.randint(0, window_days, n).tolist()]
        assignment_departments = np.random.choice(departments, n)
    print(f"Generating patient_assignment ({n:,} rows)")
    assignment_departments = np.array(assignment_departments, dtype=object)
    case_arr = np.empty(n, dtype=object)
    for dept in np.unique(assignment_departments):
        mask = (assignment_departments == dept)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        probs = dept_case_probs.get(dept, default_case_probs)
        case_arr[mask] = np.random.choice(case_types, cnt, p=probs)
    surge_arr  = np.array([1 if is_surge(d.date()) else 0 for d in dates])

    surge_emergency_flag = np.where(
        (surge_arr == 1) & (case_arr == 'Emergency'), 1, 0
    )

    df = pd.DataFrame({
        'assignment_id':              [f"ASN_{i:08d}" for i in range(1, n + 1)],
        'staff_id':                   selected,
        'patient_id':                 [f"PAT_{np.random.randint(100000, 999999)}" for _ in range(n)],
        'assignment_date':            [d.date() for d in dates],
        'severity_level':             np.random.randint(1, 6, n),
        'department':                 assignment_departments,
        'case_type':                  case_arr,
        'assignment_duration_minutes': np.random.randint(15, 181, n),
        'critical_flag':              np.random.choice([0, 1], n, p=[0.78, 0.22]),
        'surge_emergency_flag':       surge_emergency_flag,
    })

    path = os.path.join(OUTPUT_DIR, "patient_assignment.parquet")
    df.to_parquet(path, index=False)
    print(f"patient_assignment -> {path} ({time.time()-t0:.1f}s)")
    return df


# 
# 4. leave_records
# 
def generate_leave_records(
    staff_ids: np.ndarray,
    ts_df: pd.DataFrame = None,
    staff_df: pd.DataFrame = None
):
    t0  = _t()
    n   = OTHER_COUNT
    print(f"Generating leave_records ({n:,} rows)")

    leave_types = ['Sick', 'Vacation', 'Personal', 'Maternity/Paternity', 'Study', 'Bereavement']
    statuses    = ['Approved', 'Pending', 'Rejected']

    if staff_df is not None:
        eligible_staff_ids = staff_df.loc[staff_df['status'] != 'Inactive', 'staff_id'].values
    else:
        eligible_staff_ids = staff_ids

    if len(eligible_staff_ids) == 0:
        raise ValueError("No eligible staff IDs available for leave generation")

    selected = np.random.choice(eligible_staff_ids, n)
    base_date = datetime.combine(TIMESHEET_START_DATE, datetime.min.time())
    window_days = (TIMESHEET_END_DATE - TIMESHEET_START_DATE).days + 1
    starts   = [base_date + timedelta(days=int(d)) for d in np.random.randint(0, window_days, n).tolist()]
    durations = np.random.randint(1, 15, n).tolist()

    if ts_df is not None:
        # Build worked-day lookup only for staff selected in this leave batch.
        selected_unique = set(selected.tolist())
        ts_small = ts_df[ts_df['staff_id'].isin(selected_unique)][['staff_id', 'shift_date']].copy()
        ts_small['shift_date'] = pd.to_datetime(ts_small['shift_date']).dt.date
        worked_days_map = ts_small.groupby('staff_id')['shift_date'].agg(lambda x: set(x)).to_dict()

        adjusted_starts = []
        adjusted_ends = []
        for sid, sdt, dur in zip(selected, starts, durations):
            s = sdt.date()
            e = min(s + timedelta(days=int(dur)), TIMESHEET_END_DATE)
            worked_days = worked_days_map.get(sid, set())

            # Shift leave window forward if it overlaps any worked day.
            attempts = 0
            while attempts < 120 and any((s + timedelta(days=k)) in worked_days for k in range((e - s).days + 1)):
                s = min(s + timedelta(days=1), TIMESHEET_END_DATE)
                e = min(s + timedelta(days=int(dur)), TIMESHEET_END_DATE)
                attempts += 1

            adjusted_starts.append(s)
            adjusted_ends.append(e)
        starts = adjusted_starts
        ends = adjusted_ends
    else:
        ends = [(min(s.date() + timedelta(days=int(d)), TIMESHEET_END_DATE)) for s, d in zip(starts, durations)]
        starts = [s.date() for s in starts]

    df = pd.DataFrame({
        'leave_id':            [f"LV_{i:08d}" for i in range(1, n + 1)],
        'staff_id':            selected,
        'leave_start':         starts,
        'leave_end':           ends,
        'leave_type':          np.random.choice(leave_types, n),
        'leave_days_count':    [(e - s).days for s, e in zip(starts, ends)],
        'approval_status':     np.random.choice(statuses, n, p=[0.78, 0.12, 0.10]),
        'emergency_leave_flag': np.random.choice([True, False], n, p=[0.12, 0.88]),
    })

    path = os.path.join(OUTPUT_DIR, "leave_records.parquet")
    df.to_parquet(path, index=False)
    print(f"leave_records -> {path} ({time.time()-t0:.1f}s)")
    return df


# 
# 5. Back-fill night_shift_female_flag into staff_master
# 
def backfill_female_night_flag(staff_df: pd.DataFrame, ts_df: pd.DataFrame):
    t0 = _t()
    print("Back-filling night_shift_female_flag in staff_master")
    females_on_night = set(ts_df[ts_df['female_night_flag'] == 1]['staff_id'].unique())
    staff_df['night_shift_female_flag'] = staff_df['staff_id'].apply(
        lambda x: 1 if x in females_on_night else 0
    )
    path = os.path.join(OUTPUT_DIR, "staff_master.parquet")
    staff_df.to_parquet(path, index=False)
    print(f"staff_master re-saved ({time.time()-t0:.1f}s)")
    return staff_df


# 
# 6. DIMENSION TABLES
# 
def generate_dim_date():
    t0 = _t()
    print(" Generating dim_date")
    start, end = date(2024, 1, 1), date(2026, 12, 31)
    dates = pd.date_range(start, end, freq='D')
    df = pd.DataFrame({
        'date_key':    dates.strftime('%Y%m%d').astype(int),
        'full_date':   dates.date,
        'day_of_week': dates.day_name(),
        'day_num':     dates.dayofweek,          # 0=Mon  6=Sun
        'week_number': dates.isocalendar().week.values,
        'month_num':   dates.month,
        'month_name':  dates.month_name(),
        'quarter':     dates.quarter,
        'year':        dates.year,
        'is_weekend':  (dates.dayofweek >= 5).astype(int),
        'is_surge_period': [1 if is_surge(d.date()) else 0 for d in dates],
    })
    path = os.path.join(OUTPUT_DIR, "dim_date.parquet")
    df.to_parquet(path, index=False)
    print(f" dim_date   {path}  ({time.time()-t0:.1f}s)")


def generate_dim_department():
    t0 = _t()
    print(" Generating dim_department")
    depts = ['ICU', 'ER', 'General Ward', 'Pediatrics',
             'Oncology', 'Cardiology', 'Radiology', 'Pathology']
    risk_level = {
        'ICU': 'Very High', 'ER': 'Very High', 'Oncology': 'High',
        'Cardiology': 'High', 'General Ward': 'Medium', 'Pediatrics': 'Medium',
        'Radiology': 'Low', 'Pathology': 'Low'
    }
    df = pd.DataFrame({
        'dept_id':          [f"DEPT_{i+1:02d}" for i in range(len(depts))],
        'department_name':  depts,
        'risk_category':    [risk_level[d] for d in depts],
        'is_critical_dept': [1 if risk_level[d] in ('Very High', 'High') else 0 for d in depts],
    })
    path = os.path.join(OUTPUT_DIR, "dim_department.parquet")
    df.to_parquet(path, index=False)
    print(f" dim_department   {path}  ({time.time()-t0:.1f}s)")


def generate_dim_shift_type():
    t0 = _t()
    print(" Generating dim_shift_type")
    df = pd.DataFrame({
        'shift_type_id':   [1, 2],
        'shift_type_name': ['Day', 'Night'],
        'shift_start_hr':  [7, 19],
        'burnout_weight':  [1.0, 1.5],       # Night shifts weigh 50% more on burnout
        'description':     ['Standard Daytime Shift (07:0019:00)',
                            'Night / On-call Shift   (19:0007:00)']
    })
    path = os.path.join(OUTPUT_DIR, "dim_shift_type.parquet")
    df.to_parquet(path, index=False)
    print(f" dim_shift_type   {path}  ({time.time()-t0:.1f}s)")


def generate_dim_staff(staff_df: pd.DataFrame):
    t0 = _t()
    print(" Generating dim_staff")
    cols = ['staff_id', 'staff_name', 'gender', 'role', 'experience_level',
            'department', 'hire_date', 'employment_type', 'shift_preference',
            'status', 'base_location', 'per_hour_rate', 'salary_band',
            'night_shift_female_flag']
    df = staff_df[cols].copy()
    path = os.path.join(OUTPUT_DIR, "dim_staff.parquet")
    df.to_parquet(path, index=False)
    print(f" dim_staff   {path}  ({time.time()-t0:.1f}s)")


# 
# 7. FACT TABLES
# 
def generate_fact_burnout(ts_df: pd.DataFrame, staff_df: pd.DataFrame):
    """
    Grain: staff_id  ISO week
    Aggregated KPIs: total_overtime, avg_burnout_score, max_burnout_score,
                     night_shifts, surge_shifts, high_burnout_days,
                     consecutive_days_flag, back_to_back_count,
                     burnout_risk_label (worst in week)
    """
    t0 = _t()
    print(" Generating fact_burnout")

    ts = ts_df.copy()
    ts['shift_date_dt'] = pd.to_datetime(ts['shift_date'])
    ts['year_week']     = ts['shift_date_dt'].dt.strftime('%G-W%V')   # ISO week

    risk_order = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    ts['risk_num'] = ts['burnout_risk_label'].map(risk_order)

    ts['night_shift_flag'] = (ts['shift_type'] == 'Night').astype(int)

    grp = ts.groupby(['staff_id', 'year_week']).agg(
        total_overtime_hours        = ('overtime_hours',         'sum'),
        avg_burnout_score           = ('burnout_score',          'mean'),
        max_burnout_score           = ('burnout_score',          'max'),
        night_shifts_count          = ('night_shift_flag',       'sum'),
        surge_shifts_count          = ('surge_period_flag',      'sum'),
        high_burnout_days           = ('burnout_score',          lambda x: (x >= 76).sum()),
        consecutive_days_flag       = ('consecutive_days_flag',  'max'),
        back_to_back_12h_count      = ('back_to_back_12h_flag',  'sum'),
        night_to_day_flips          = ('night_to_day_flip_flag', 'sum'),
        female_night_shifts         = ('female_night_flag',      'sum'),
        worst_risk_num              = ('risk_num',               'max'),
        total_shift_hours           = ('shift_duration_hours',   'sum'),
        shift_count                 = ('timesheet_id',           'count'),
    ).reset_index()

    risk_reverse = {v: k for k, v in risk_order.items()}
    grp['worst_burnout_risk'] = grp['worst_risk_num'].map(risk_reverse)
    grp.drop(columns=['worst_risk_num'], inplace=True)

    # Join salary band and role for analysis
    staff_small = staff_df[['staff_id', 'role', 'salary_band', 'gender', 'department']].drop_duplicates('staff_id')
    grp = grp.merge(staff_small, on='staff_id', how='left')

    # Fact key
    grp.insert(0, 'burnout_fact_id', [f"BF_{i:08d}" for i in range(1, len(grp) + 1)])

    # Round floats
    for c in ['total_overtime_hours', 'avg_burnout_score']:
        grp[c] = grp[c].round(2)

    path = os.path.join(OUTPUT_DIR, "fact_burnout.parquet")
    grp.to_parquet(path, index=False)
    print(f" fact_burnout   {path}  ({time.time()-t0:.1f}s)  [{len(grp):,} rows]")


def generate_fact_overtime(ts_df: pd.DataFrame):
    """
    Grain: one row per timesheet record where overtime_hours > 0
    Answer: 'Who works excessive overtime?'
    """
    t0 = _t()
    print("Generating fact_overtime")

    df = ts_df[ts_df['overtime_hours'] > 0][[
        'timesheet_id', 'staff_id', 'shift_date', 'shift_type',
        'overtime_hours', 'shift_duration_hours', 'burnout_score',
        'burnout_risk_label', 'surge_period_flag', 'consecutive_days_flag',
        'back_to_back_12h_flag', 'night_to_day_flip_flag', 'female_night_flag'
    ]].copy()

    df.insert(0, 'overtime_fact_id', [f"OT_{i:08d}" for i in range(1, len(df) + 1)])

    path = os.path.join(OUTPUT_DIR, "fact_overtime.parquet")
    df.to_parquet(path, index=False)
    print(f" fact_overtime   {path}  ({time.time()-t0:.1f}s)  [{len(df):,} rows]")


def generate_fact_leave(leave_df: pd.DataFrame):
    """Grain: leave record. Includes derived burnout pressure indicator."""
    t0 = _t()
    print("Generating fact_leave")

    df = leave_df.copy()
    # High-risk leave: emergency + long duration
    df['burnout_leave_flag'] = (
        (df['emergency_leave_flag'] == True) | (df['leave_days_count'] >= 10)
    ).astype(int)

    df.insert(0, 'leave_fact_id', [f"LF_{i:08d}" for i in range(1, len(df) + 1)])

    path = os.path.join(OUTPUT_DIR, "fact_leave.parquet")
    df.to_parquet(path, index=False)
    print(f" fact_leave   {path}  ({time.time()-t0:.1f}s)  [{len(df):,} rows]")


# 
# MAIN    Only 4 tables are produced
# 
if __name__ == "__main__":
    T0 = _t()
    np.random.seed(SEED)

    print("\n" + "="*65)
    print("Staff Optimization & Burnout Prevention - Data Generator v3")
    print("OUTPUT: 4 enriched tables only")
    print("="*65 + "\n")

    #  Step 1: staff_master 
    staff_df = generate_staff_master()

    #  Step 2: staff_timesheet (auto-generates linked patient_assignment if not passed) 
    #
    #   burnout_score = (overtime_hours  2)
    #                 + (critical_cases  3)    from patient_assignment
    #                 + (night_shift     5)    1 if Night shift
    #                 + (consecutive_days  4)  1 if >4 consecutive days
    #                 + (emergency_cases  3)   from patient_assignment
    #
    ts_df = generate_staff_timesheet(staff_df, pa_df=None)

    #  Step 3: leave_records (aligned to avoid timesheet overlap) 
    lv_df = generate_leave_records(staff_df['staff_id'].values, ts_df=ts_df, staff_df=staff_df)

    #  Back-fill female night flag into staff_master 
    staff_df = backfill_female_night_flag(staff_df, ts_df)

    print("\n" + "="*65)
    print(f"All done in {time.time()-T0:.1f}s")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print("="*65 + "\n")

    #  Quick QA summary 
    CORE_FILES = ['staff_master.parquet', 'staff_timesheet.parquet',
                  'patient_assignment.parquet', 'leave_records.parquet']
    print("QA Summary - 4 Tables")
    print("-"*55)
    for f in CORE_FILES:
        p   = os.path.join(OUTPUT_DIR, f)
        tmp = pd.read_parquet(p)
        mb  = os.path.getsize(p) / 1e6
        print(f"  {f:<35} {len(tmp):>10,} rows  {mb:6.1f} MB")
        print(f"  {'Columns:':<35} {list(tmp.columns)}")
        print()
