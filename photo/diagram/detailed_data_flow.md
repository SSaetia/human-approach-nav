%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
flowchart TD
    %% Node Styles
    classDef node fill:#fff,stroke:#333,stroke-width:2px;
    classDef topic fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 2;

    %% --- NODES ---
    subgraph PERCEPTION ["Perception Source"]
        HR("human_replay.py"):::node
    end

    subgraph PROCESSING ["Data Processing"]
        HOC("human_obstacle_cloud.py"):::node
        AGS("approach_goal_sender.py"):::node
    end

    subgraph NAV2 ["Nav2 System"]
        BTN("bt_navigator"):::node
        CTRL("controller_server"):::node
        LC("Costmap (ObstacleLayer)"):::node
    end

    subgraph SIM ["Simulation Feedback"]
        FO("fake_odom.py"):::node
    end

    %% --- DATA FLOW & TOPICS ---
    
    %% Human Pose Flow
    HR -- "/human/pose<br>(frame: map)" --> HOC
    HR -- "/human/pose<br>(frame: map)" --> AGS

    %% Obstacle Logic
    HOC -- "/human/obstacles<br>(PointCloud2, frame: map)" --> LC
    LC -- "Mark Obstacles" --> CTRL

    %% Decision Logic
    AGS -- "Action: NavigateToPose<br>(Target: approach_pose)" --> BTN
    AGS -- "/speed_limit" --> CTRL

    %% Navigation Execution
    BTN -- "FollowPath" --> CTRL
    CTRL -- "/cmd_vel" --> FO

    %% Localization Loop
    FO -- "/odom & /tf<br>(odom->base_link)" --> LC
    FO -- "/odom & /tf<br>(odom->base_link)" --> AGS
    FO -- "/odom & /tf<br>(odom->base_link)" --> BTN

    %% Map
    MAP[("Map Server")] -- "/map" --> BTN
    MAP[("Map Server")] -- "/map" --> LC