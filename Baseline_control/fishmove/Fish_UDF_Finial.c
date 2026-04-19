#include "udf.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#define L       1                 
#define PI      M_PI             
#define A_s     (0.47 * PI / L)
#define A_w     (1.4)
#define phi_0   (0.0 * PI)
#define A_0     (0.5)
#define A_1     (-0.2)
#define lambda  L
#define x_h     (0.1 * L)
#define L_max   1
#define N       1200

#define mass 0.8
#define I 0.5

#define NUM_SENSORS 10
#define SENSOR_NOT_FOUND -1
#define SENSOR_TOLERANCE 0.05
#define COORD_OFFSET 10000.0
#define VALUE_OFFSET 100000000.0

// === 控制数组和变量 ===
real direction_array[40];
real time_array[40];
int direction_count = 0;
int time_count = 0;
int array_length = sizeof(direction_array) / sizeof(direction_array[0]);

real xdisplacement = 0.0, xvelocity = 0.0, xacceleration = 0.0;
real ydisplacement = 0.0, yvelocity = 0.0, yacceleration = 0.0;
real thetadisplacement = 0.0, thetavelocity = 0.0, thetaacceleration = 0.0;

real phi_new[N] = { 0.0 };
real S_new[N] = { 0.0 };
real x_new[N] = { 0.0 };
real y_new[N] = { 0.0 };

real sensor_relative_x[NUM_SENSORS] = { 0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0 };
real sensor_relative_y[NUM_SENSORS] = { 0, 0.06, -0.06, 0.06, -0.06, 0.05, -0.05, 0.03, -0.03, 0 };
face_t sensor_face_ids[NUM_SENSORS];
int sensors_initialized = 0;

real sensor_current_x[NUM_SENSORS]; 
real sensor_current_y[NUM_SENSORS];

int need_rebuild_nodes = 0;

void generate_state_filename(char* filename, real current_time)
{
    sprintf(filename, "fish_state_t%.6f.txt", current_time);
}

// === 控制方向获取函数 ===
real get_direction(real t)
{
	real cumulative_time = 0.0;
	real total_cycle_time = 0.0;
	for (int i = 0; i < direction_count; i++)
		total_cycle_time += time_array[i];
	if (total_cycle_time == 0.0) return 0.0;
	real cycle_time = fmod(t, total_cycle_time);
	for (int i = 0; i < direction_count; i++)
	{
		if (cycle_time <= cumulative_time + time_array[i])
			return direction_array[i];
		cumulative_time += time_array[i];
	}
	return 0.0;
}

// === tau 函数 ===
real tau(real t)
{
	real T_c = RP_Get_Real("tc");;
	real T_s = 0.2 * T_c;
	if (t <= T_s)
		return (t / T_s - (1 / (2 * PI)) * sin(2 * PI * t / T_s));
	else
		return 1.0;
}

// === k 函数 ===
real k(real s, real t)
{
	real T_c = RP_Get_Real("tc");;
	real T_w = T_c;
	if (s <= x_h) return 0.0;
	real current_direction = get_direction(t);
	real turning_term = (current_direction == 0.0) ? 0.0 : current_direction * A_s * (1 - cos(2 * PI * t / T_c));
	return turning_term + A_w * (pow(s / L, 2) + A_1 * s / L + A_0) * tau(t) * sin(2 * PI * (s / lambda - t / T_w) + phi_0);
}

// === 计算中线坐标 ===
void calculate_s_cor(real t, real* S, real* phi, real* s_x, real* s_y)
{
	for (int i = 1; i < N; i++)
	{
		real ds = 0.001;
		S[i] = S[i - 1] + ds;
		phi[i] = phi[i - 1] + k(S[i], t) * ds;
		s_x[i] = s_x[i - 1] + cos(phi[i]) * ds;
		s_y[i] = s_y[i - 1] + sin(phi[i]) * ds;
	}
}

// === 添加控制动作 ===
void add_direction_value(real A_t)
{
	if (direction_count < array_length)
		direction_array[direction_count++] = A_t;
}

void add_period_value(real T_c)
{
	if (time_count < array_length)
		time_array[time_count++] = T_c;
}

DEFINE_ON_DEMAND(add_action_from_console)
{
	real A_t_val = RP_Get_Real("at");
	real T_c_val = RP_Get_Real("tc");
	if (direction_count < array_length && time_count < array_length)
	{
		add_direction_value(A_t_val);
		add_period_value(T_c_val);
		Message("\n Add new action: A_t = %.3f, T_c = %.3f\n", A_t_val, T_c_val);

		// Output the current direction_array and time_array to the console
		Message("\nCurrent direction array: ");
		for (int i = 0; i < direction_count; i++)
		{
			Message("%.3f ", direction_array[i]);
		}
		Message("\n");

		Message("Current time array: ");
		for (int i = 0; i < time_count; i++)
		{
			Message("%.3f ", time_array[i]);
		}
		Message("\n");
	}
	else
	{
		Message("\n Action buffer full. Max entries = %d\n", array_length);
	}
}


void update_sensor_positions()
{
    int i;
    for (i = 0; i < NUM_SENSORS; i++)
    {
        sensor_current_x[i] = sensor_relative_x[i] * cos(thetadisplacement) - sensor_relative_y[i] * sin(thetadisplacement) + xdisplacement;
        sensor_current_y[i] = sensor_relative_x[i] * sin(thetadisplacement) + sensor_relative_y[i] * cos(thetadisplacement) + ydisplacement;
    }
}

face_t find_closest_face_to_sensor(Thread* tf, real target_x, real target_y)
{
    face_t f, closest_f = SENSOR_NOT_FOUND;
    real min_distance = 1e10;
    real centroid[ND_ND];
    real face_x, face_y;

    if (tf == NULL)
    {
        return SENSOR_NOT_FOUND;
    }

    begin_f_loop(f, tf)
    {
        F_CENTROID(centroid, f, tf);
        face_x = centroid[0];
        face_y = centroid[1];

        real distance = sqrt(pow(face_x - target_x, 2) + pow(face_y - target_y, 2));

        if (distance < min_distance)
        {
            min_distance = distance;
            closest_f = f;
        }
    }
    end_f_loop(f, tf);

    if (min_distance < SENSOR_TOLERANCE)
    {
        return closest_f;
    }
    else
    {
        return SENSOR_NOT_FOUND;
    }
}

real get_sensor_pressure(Thread* tf, face_t face_id)
{
    if (tf == NULL || face_id == SENSOR_NOT_FOUND)
    {
        return -1.0;
    }

    return F_P(face_id, tf) + VALUE_OFFSET;
}

real get_sensor_ux(Thread* tf, face_t face_id)
{
    if (tf == NULL || face_id == SENSOR_NOT_FOUND)
    {
        return -1.0;
    }

    return F_U(face_id, tf) + VALUE_OFFSET;
}

real get_sensor_uy(Thread* tf, face_t face_id)
{
    if (tf == NULL || face_id == SENSOR_NOT_FOUND)
    {
        return -1.0;
    }

    return F_V(face_id, tf) + VALUE_OFFSET;
}

DEFINE_ON_DEMAND(save_fish_state)
{
    char sim_time_filename[256];
    FILE* fp = NULL;
    FILE* fp_latest = NULL;
    int i;

    generate_state_filename(sim_time_filename, CURRENT_TIME);

    update_sensor_positions();

    node_to_host_real(sensor_current_x, NUM_SENSORS);
    node_to_host_real(sensor_current_y, NUM_SENSORS);

#if RP_HOST
    fp = fopen(sim_time_filename, "w");
    if (fp == NULL)
    {
        Message("Error: Cannot create state file %s\n", sim_time_filename);
        return;
    }
    
    fprintf(fp, "XDISP %.16e\n", xdisplacement);
    fprintf(fp, "YDISP %.16e\n", ydisplacement);
    fprintf(fp, "THETADISP %.16e\n", thetadisplacement);
    fprintf(fp, "XVEL %.16e\n", xvelocity);
    fprintf(fp, "YVEL %.16e\n", yvelocity);
    fprintf(fp, "THETAVEL %.16e\n", thetavelocity);
    fprintf(fp, "XACCEL %.16e\n", xacceleration);
    fprintf(fp, "YACCEL %.16e\n", yacceleration);
    fprintf(fp, "THETAACCEL %.16e\n", thetaacceleration);
    
    fprintf(fp, "SENSOR_COORDS_X");
    for (i = 0; i < NUM_SENSORS; i++)
    {
        fprintf(fp, " %.16e", sensor_current_x[i]);
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "SENSOR_COORDS_Y");
    for (i = 0; i < NUM_SENSORS; i++)
    {
        fprintf(fp, " %.16e", sensor_current_y[i]);
    }
    fprintf(fp, "\n");
    
    fclose(fp);

    fp_latest = fopen("fish_latest_state.txt", "w");
    if (fp_latest != NULL)
    {
        fprintf(fp_latest, "%s\n", sim_time_filename);
        fclose(fp_latest);
    }
    
    Message("Fish state saved to: %s\n", sim_time_filename);
#endif
}

DEFINE_ON_DEMAND(load_fish_state)
{
    char filename_to_load[256];
    FILE* fp_latest = NULL;
    FILE* fp = NULL;
    char line[256];
    char keyword[100];
    
#if RP_HOST
    fp_latest = fopen("fish_latest_state.txt", "r");
    if (fp_latest != NULL)
    {
        fscanf(fp_latest, "%s", filename_to_load);
        fclose(fp_latest);
    }
    else
    {
        strcpy(filename_to_load, "fish_restart_state.txt");
    }
    
    fp = fopen(filename_to_load, "r");
    if (fp == NULL)
    {
        Message("Warning: State file not found, using default values\n");
        return;
    }
    
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (line[0] == '#') continue;
        
        if (sscanf(line, "%s", keyword) == 1)
        {
            if (strcmp(keyword, "XDISP") == 0)
            {
                sscanf(line, "%s %lf", keyword, &xdisplacement);
            }
            else if (strcmp(keyword, "YDISP") == 0)
            {
                sscanf(line, "%s %lf", keyword, &ydisplacement);
            }
            else if (strcmp(keyword, "THETADISP") == 0)
            {
                sscanf(line, "%s %lf", keyword, &thetadisplacement);
            }
            else if (strcmp(keyword, "XVEL") == 0)
            {
                sscanf(line, "%s %lf", keyword, &xvelocity);
            }
            else if (strcmp(keyword, "YVEL") == 0)
            {
                sscanf(line, "%s %lf", keyword, &yvelocity);
            }
            else if (strcmp(keyword, "THETAVEL") == 0)
            {
                sscanf(line, "%s %lf", keyword, &thetavelocity);
            }
            else if (strcmp(keyword, "XACCEL") == 0)
            {
                sscanf(line, "%s %lf", keyword, &xacceleration);
            }
            else if (strcmp(keyword, "YACCEL") == 0)
            {
                sscanf(line, "%s %lf", keyword, &yacceleration);
            }
            else if (strcmp(keyword, "THETAACCEL") == 0)
            {
                sscanf(line, "%s %lf", keyword, &thetaacceleration);
            }
        }
    }
    
    fclose(fp);
#endif
    
    host_to_node_real_3(xdisplacement, ydisplacement, thetadisplacement);
    host_to_node_real_3(xvelocity, yvelocity, thetavelocity);
    host_to_node_real_3(xacceleration, yacceleration, thetaacceleration);
    
    need_rebuild_nodes = 1;
    
#if RP_HOST
    Message("Fish state loaded from: %s\n", filename_to_load);
    Message("Position: (%.6f, %.6f, %.6f deg)\n", xdisplacement, ydisplacement, thetadisplacement*180/PI);
#endif
}

DEFINE_ON_DEMAND(initialize_sensors)
{
    Domain* domain = Get_Domain(1);
    Thread* tf = Lookup_Thread(domain, 11);

    if (tf == NULL)
    {
        Message("Error: Could not find fish surface thread (ID: 11)\n");
        return;
    }

    update_sensor_positions();

    if (I_AM_NODE_ZERO_P)
    {
        Message("Sensor system initialized with dynamic positioning.\n");
    }
}

DEFINE_EXECUTE_AT_END(execute_at_end)
{
    real x_cg[3], f_glob[3], m_glob[3];
    Domain* domain = Get_Domain(1);
    Thread* tf = Lookup_Thread(domain, 11);
    FILE* fp = NULL;
    FILE* fp_p = NULL;
    FILE* fp_ux = NULL;
    FILE* fp_uy = NULL;
    real pressures[NUM_SENSORS];
    real xvels[NUM_SENSORS];
    real yvels[NUM_SENSORS];
    int i;
    
    x_cg[0] = xdisplacement + 0.42 * L;
    x_cg[1] = ydisplacement;

    if (tf != NULL)
    {
        Compute_Force_And_Moment(domain, tf, x_cg, f_glob, m_glob, TRUE);
    }
    else
    {
        f_glob[0] = f_glob[1] = f_glob[2] = 0.0;
        m_glob[0] = m_glob[1] = m_glob[2] = 0.0;
    }

    real xvelocity_before = xvelocity, xacceleration_before = xacceleration;
    xacceleration = f_glob[0] / mass;
    xvelocity += (xacceleration + xacceleration_before) * CURRENT_TIMESTEP / 2;
    xdisplacement += (xvelocity_before + xvelocity) * CURRENT_TIMESTEP / 2;

    real yvelocity_before = yvelocity, yacceleration_before = yacceleration;
    yacceleration = f_glob[1] / mass;
    yvelocity += (yacceleration + yacceleration_before) * CURRENT_TIMESTEP / 2;
    ydisplacement += (yvelocity_before + yvelocity) * CURRENT_TIMESTEP / 2;

    real thetavelocity_before = thetavelocity, thetaacceleration_before = thetaacceleration;
    thetaacceleration = m_glob[2] / I;
    thetavelocity += (thetaacceleration + thetaacceleration_before) * CURRENT_TIMESTEP / 2;
    thetadisplacement += (thetavelocity_before + thetavelocity) * CURRENT_TIMESTEP / 2;

#if RP_HOST
    fp = fopen("Output.txt", "a");
    if (fp != NULL)
    {
        fprintf(fp, "%.4e %.4e %.4e %.4e %.4e %.4e %.4e %.4e", CURRENT_TIME, xdisplacement, ydisplacement, thetadisplacement, f_glob[0], f_glob[1], m_glob[2], get_direction(CURRENT_TIME));
        fprintf(fp, "\n");
        fclose(fp);
    }
#endif

    update_sensor_positions();

#if RP_HOST
    fp_p = fopen("Pressure_Sensors.txt", "a");
    fp_ux = fopen("Xvelocity_Sensors.txt", "a");
    fp_uy = fopen("Yvelocity_Sensors.txt", "a");

    if (fp_p != NULL) fprintf(fp_p, "%.4e", CURRENT_TIME);
    if (fp_ux != NULL) fprintf(fp_ux, "%.4e", CURRENT_TIME);
    if (fp_uy != NULL) fprintf(fp_uy, "%.4e", CURRENT_TIME);
#endif

    for (i = 0; i < NUM_SENSORS; i++)
    {
        real local_p = -1.0, local_ux = -1.0, local_uy = -1.0;
        
        if (tf != NULL)
        {
            face_t current_face = find_closest_face_to_sensor(tf, sensor_current_x[i], sensor_current_y[i]);
            
            local_p = get_sensor_pressure(tf, current_face);
            local_ux = get_sensor_ux(tf, current_face);
            local_uy = get_sensor_uy(tf, current_face);
        }

        pressures[i] = PRF_GRHIGH1(local_p);
        if (pressures[i] >= 0.0) {
            pressures[i] -= VALUE_OFFSET;
        } else {
            pressures[i] = 999.0;
        }

        xvels[i] = PRF_GRHIGH1(local_ux);
        if (xvels[i] >= 0.0) {
            xvels[i] -= VALUE_OFFSET;
        } else {
            xvels[i] = 999.0;
        }

        yvels[i] = PRF_GRHIGH1(local_uy);
        if (yvels[i] >= 0.0) {
            yvels[i] -= VALUE_OFFSET;
        } else {
            yvels[i] = 999.0;
        }
    }

    node_to_host_real(pressures, NUM_SENSORS);
    node_to_host_real(xvels, NUM_SENSORS);
    node_to_host_real(yvels, NUM_SENSORS);

#if RP_HOST
    for (i = 0; i < NUM_SENSORS; i++)
    {
        if (fp_p != NULL) fprintf(fp_p, " %.4e", pressures[i]);
        if (fp_ux != NULL) fprintf(fp_ux, " %.4e", xvels[i]);
        if (fp_uy != NULL) fprintf(fp_uy, " %.4e", yvels[i]);
    }
    if (fp_p != NULL) fprintf(fp_p, "\n");
    if (fp_ux != NULL) fprintf(fp_ux, "\n");
    if (fp_uy != NULL) fprintf(fp_uy, "\n");

    if (fp_p != NULL) fclose(fp_p);
    if (fp_ux != NULL) fclose(fp_ux);
    if (fp_uy != NULL) fclose(fp_uy);
#endif
}

DEFINE_ZONE_MOTION(zone, omega, axis, origin, velocity, time, dtime)
{
    velocity[0] = xvelocity;
    velocity[1] = yvelocity;
    origin[0] = xdisplacement;
    origin[1] = ydisplacement;
    *omega = thetavelocity;
}

DEFINE_GRID_MOTION(fish, domain, dt, time, dtime)
{
    face_t f;
    Thread* tf = DT_THREAD(dt);
    real NV_VEC(coordinate);
    int n;
    Node* v;
    real x, y, rx, ry;
    int i;
    int rebuild_done = 0;
    
    calculate_s_cor(time, S_new, phi_new, x_new, y_new);

    SET_DEFORMING_THREAD_FLAG(THREAD_T0(tf));

    begin_f_loop(f, tf)
    {
        f_node_loop(f, tf, n)
        {
            v = F_NODE(f, tf, n);
            x = NODE_X(v);
            y = NODE_Y(v);

            if (NODE_POS_NEED_UPDATE(v))
            {
                NODE_POS_UPDATED(v);

                if (time <= dtime )
                {
                    real global_x = x;
                    real global_y = y;
                    real local_x = (global_x - xdisplacement) * cos(thetadisplacement) + (global_y - ydisplacement) * sin(thetadisplacement);
                    real local_y = -(global_x - xdisplacement) * sin(thetadisplacement) + (global_y - ydisplacement) * cos(thetadisplacement);
                    if (local_x < 0.0)
                    {
                        local_x = 0.0;
                    }
                    N_UDMI(v, 0) = local_x;
                    N_UDMI(v, 1) = local_y;
                 
                    coordinate[0] = global_x;
                    coordinate[1] = global_y;
                    rebuild_done = 1;
                }
                else
                {
                    real node_l = N_UDMI(v, 0);
                    real node_width = N_UDMI(v, 1);

                    int found = 0;
                    for (i = 1; i < N; i++)
                    {
                        if (S_new[i - 1] <= node_l && node_l <= S_new[i])
                        {
                            rx = x_new[i] - node_width * sin(phi_new[i]);
                            ry = y_new[i] + node_width * cos(phi_new[i]);

                            coordinate[0] = rx * cos(thetadisplacement + thetavelocity * dtime) - ry * sin(thetadisplacement + thetavelocity * dtime) + xdisplacement + xvelocity * dtime;
                            coordinate[1] = rx * sin(thetadisplacement + thetavelocity * dtime) + ry * cos(thetadisplacement + thetavelocity * dtime) + ydisplacement + yvelocity * dtime;
                            found = 1;
                            break;
                        }
                    }
                    if (!found)
                    {
                        coordinate[0] = x;
                        coordinate[1] = y;
                    }
                }

                NV_V(NODE_COORD(v), =, coordinate);
            }
        }
    }
    end_f_loop(f, tf);

    if (rebuild_done && need_rebuild_nodes == 1)
    {
        need_rebuild_nodes = 0;
    }
}