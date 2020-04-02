

from nptdms import TdmsFile
 
from pixels import ioutils


label_map = {
    1: 'uncued_push',
    2: 'rewarded_push',
    3: 'missed_push',
    4: 'correct_rejection',
    5: 'false_alarm',
}


def get_action_labels(data, to_freq=None, from_freq=None):

    if to_freq is None:  # this should be taken from neuropixels metadata
        to_freq = 1000
    if from_freq is None:  # this should be taken from metadata
        from_freq = 1000

    frontch = 1;
    backch = 2;
    maxRT = 2 * freq  # 2s max RT for lever push

    resample_factor = to_freq / from_freq  # find downsampling factor from behaviour to imaging rate

    back_sensor_on = numpy.where(data['Back_Sensor'] > 2)
    front_sensor_on = numpy.where(data['Front_Sensor'] > 2)
    front_sensor_off = numpy.where(data['Front_Sensor'] < 2)
    reward_signal = numpy.where(data['Reward_Signal'] < 2)


    Rewards = find(Leverabfdata(delay:end-1,3)<2 & Leverabfdata(delay+1:end,3)>2); %Find all reward times
    EndRewards = find(Leverabfdata(delay:end-1,3)>2 & Leverabfdata(delay+1:end,3)<=2);
    Tone_on = find(Leverabfdata(delay:end,5)>2);
    %Find push and pull onsets occurring during toneON (should exclude pre-emptives)
    
    % divide tones into push and pull tones
    
    pushToneTimes= find(Leverabfdata(delay:end-1,5)<2 & Leverabfdata(delay+1:end,5)>2 & Leverabfdata(delay+1:end,backch)<0.5); %find times when tone come on and lever is at back
    pullToneTimes = find(Leverabfdata(delay:end-1,5)<2 & Leverabfdata(delay+1:end,5)>2 & Leverabfdata(delay+1:end,frontch)<0.5); %find times when tone come on and lever is at front
    AllToneTimes = find(Leverabfdata(delay:end-1,5)<2 & Leverabfdata(delay+1:end,5)>2);
    goToneTimes = [];
    nogoToneTimes = [];
    
    
    % further subdivide pushtones into go/nogo
    
    for t = 1:length(pushToneTimes)
        if AllToneTimes(end)>pushToneTimes(t)% check for nogos only until no subsequent tone exists
            if AllToneTimes(find(AllToneTimes>pushToneTimes(t),1,'first'))-pushToneTimes(t)<1*fs_abf % if 2 tones start within 1 sec is a NoGo
                nogoToneTimes = [nogoToneTimes pushToneTimes(t)];
                AllToneTimes(find(AllToneTimes>pushToneTimes(t),1,'first'))=0; % avoid double counting NoGo tones (shaped ON-OFF-ON)
            else % is a Go
                goToneTimes = [goToneTimes pushToneTimes(t)];
            end
        elseif length(Leverabfdata(delay:end-1,5)) - pushToneTimes(t)>1*fs_abf% last one could still be a go tone
            goToneTimes = [goToneTimes pushToneTimes(t)];
        end
    end
    
    pushToneTimes = (goToneTimes(find(goToneTimes)))'; % overwrite pushToneTimes, now excluding no go tones
    AllToneTimes = AllToneTimes(find(AllToneTimes));
    
    
    
    
    %Find rewarded pushes and missed pushes
    rewardedPushes = [];
    missedPushTones = [];
    n =1;
    k = 1;
    for ipush = 1:length(pushToneTimes)
        % find next reward
        test = Rewards -  pushToneTimes(ipush,1);
        testmove = backLeverOn-pushToneTimes(ipush,1);
        Idx = find(test>0,1);
        Idxmove = find(testmove>0,1);
        % See if reward happens before next tone
        Inexttone = find(AllToneTimes > pushToneTimes(ipush),1) ;
        if isempty(Idx)
            missedPushTones(k,1) = pushToneTimes(ipush,1);
            k = k+1;
        elseif isempty(Inexttone)
    
            if backLeverOn(Idxmove)-pushToneTimes(ipush)<=maxRT % check if rt meets criterion of behaviour
                rewardedPushes(n,1) = Rewards(Idx,1);
                n = n+1;
            end
    
        elseif Rewards(Idx) < AllToneTimes(Inexttone) %
    
            if backLeverOn(Idxmove)-pushToneTimes(ipush)<=maxRT % check if rt meets criterion of behaviour
                rewardedPushes(n,1) = Rewards(Idx,1);
                n = n+1;
            end
        else
            missedPushTones(k,1) = pushToneTimes(ipush,1);
            k = k+1;
        end
    end
    
    
    %Find rewarded pulls and missed pulls
    rewardedPulls = [];
    missedPullTones = [];
    n = 1;
    k = 1;
    for ipull = 1:length(pullToneTimes)
        test =Rewards -  pullToneTimes(ipull,1);
        Idx = find(test>0,1);
        % See if reward happens before next tone
        Inexttone = find(AllToneTimes > pullToneTimes(ipull),1);
        if isempty(Idx)
            missedPullTones(k,1) = pullToneTimes(ipull,1);
            k = k+1;
        elseif isempty(Inexttone)
            rewardedPulls(n,1) = Rewards(Idx,1);
            n = n+1;
        elseif Rewards(Idx) < AllToneTimes(Inexttone)
            rewardedPulls(n,1) = Rewards(Idx,1);
            n = n+1;
        else
            missedPullTones(k,1) = pullToneTimes(ipull,1);
            k = k+1;
        end
    end
    
    
    %Find correct rejections and false alarms
    CorrectRejections = [];
    FalseAlarms = [];
    n = 1;
    k = 1;
    
    for i = 1:length(nogoToneTimes)
        % find next reward
        test = Rewards -  nogoToneTimes(i);
        Idx = find(test>0,1); % find the first reward that happens after the ith NoGo tone
        % See if reward happens before next tone
        Inexttone = find(AllToneTimes > nogoToneTimes(i),1);
        if isempty(Idx)
            FalseAlarms(k) = nogoToneTimes(i);
            k = k+1;
        elseif isempty(Inexttone)
            CorrectRejections(n) = Rewards(Idx);
            n = n+1;
        elseif Rewards(Idx) < AllToneTimes(Inexttone) %
            CorrectRejections(n) = Rewards(Idx);
            n = n+1;
        else
            FalseAlarms(k) = nogoToneTimes(i);
            k = k+1;
        end
    end
    
    % find Preemptives
    
    
    if backch == 1 % if backsensor is in ch1 assume new arduino code with resets after each tone
        Reset = find(Leverabfdata(delay:end-1,4)<2 & Leverabfdata(delay+1:end,4)>2); %currently looks for all resets
        ToneReset = [];
        for ires = 1:length(AllToneTimes)
            % find next reset
            test = Reset -  AllToneTimes(ires);
            ResIdx = find(test>0,1); % find the first reset that happens after the ith tone
            ToneReset = [ToneReset Reset(ResIdx)];
        end
        UncuedReset = setxor(Reset,ToneReset);
        PushReset = UncuedReset;
        PullReset = [];
    else % if backch is not on ch1, old arduino code was used, reset signal only after missed tone
    
        if strcmp(groupNames,'pushpull')
            PushReset = find(Leverabfdata(delay+10:end-1,6)<2 & Leverabfdata(delay+11:end,6)>2 & Leverabfdata(delay:end-11,5)<0.5);
            PullReset = find(Leverabfdata(delay+10:end-1,4)<2 & Leverabfdata(delay+11:end,4)>2 & Leverabfdata(delay:end-11,5)<0.5);
        else
            PushReset = find(Leverabfdata(delay+10:end-1,4)<2 & Leverabfdata(delay+11:end,4)>2 & Leverabfdata(delay:end-11,5)<0.5); %currently looks for all resets when tone is off to exclude resets after missed tones
            PullReset = [];
        end
    end
    
    
    % plots for debugging
    
    % figure;
    % for ch = 1:5
    %     subplot(5,1,ch)
    %
    %     plot(Leverabfdata(delay:end,ch))
    %     hold on
    %     if ch ==1||ch==2
    %         plot(PushReset,zeros(size(PushReset)),'g*')
    %         plot(Rewards,zeros(size(Rewards)),'r*')
    %     elseif ch == 4
    %         plot(PushReset,zeros(size(PushReset)),'g*')
    %     elseif ch == 3
    %         plot(Rewards,zeros(size(Rewards)),'r*')
    %     elseif ch ==5
    %         plot(Tone_on, ones(size(Tone_on)),'k*')
    %     end
    % end
    
    
    % correct for imaging sampling offset
    rewardedPushes = round(rewardedPushes./correctionf);
    rewardedPulls = round(rewardedPulls./correctionf);
    missedPushTones = round(missedPushTones./correctionf);
    missedPullTones = round(missedPullTones./correctionf);
    PushReset = round(PushReset./correctionf);
    PullReset = round(PullReset./correctionf);
    CorrectRejections = round(CorrectRejections./correctionf);
    FalseAlarms = round(FalseAlarms./correctionf);
    backLeverOn = round(backLeverOn./correctionf);
    frontLeverOn = round(frontLeverOn./correctionf);
    frontLeverOff = round(frontLeverOff./correctionf);
    Tone_on = round(Tone_on./correctionf);
    
    % plot for debugging
    
    
    
    
    % Preserve Events in downsampled output
    
    % preallocate right size
    actionLabels = zeros(1,round((size(Leverabfdata,1)-delay)/sampleDown));
    if strcmp(groupNames,'pushpull') % for push pull initialize frontllever as zeros otherwise as ones, that way it is easier to preseve salinet changes in downsampled vector
        frontLever = zeros(1,round((size(Leverabfdata,1)-delay)/sampleDown));
    else
        frontLever = ones(1,round((size(Leverabfdata,1)-delay)/sampleDown));
    end
    Tonestim = zeros(1,round((size(Leverabfdata,1)-delay)/sampleDown));
    backLever = zeros(1,round((size(Leverabfdata,1)-delay)/sampleDown));
    t_steps = zeros(1,round((size(Leverabfdata,1)-delay)/sampleDown));
    
    % Time vector
    t_steps = (1/fs:1/fs:length(t_steps)/fs);
    
    % fill in depending on Events
    backLever(ceil(backLeverOn./sampleDown)) = 1;
    if strcmp(groupNames,'pushpull')
        frontLever(ceil(frontLeverOn./sampleDown)) = 1;
    else
        frontLever(ceil(frontLeverOff./sampleDown)) = 0;
    end
    
    Tonestim(ceil(Tone_on./sampleDown)) = 1;
    
    
    % action labels
    actionLabels(ceil(PushReset./sampleDown)) = 1; % 1 for preempt push (UNCUED PUSH)
    actionLabels(ceil(rewardedPushes./sampleDown)) = 2; % 2 for rewarded push (HIT)
    actionLabels(ceil(PullReset./sampleDown)) = 3; % 3 for preemptpull
    actionLabels(ceil(rewardedPulls./sampleDown)) = 4; % 4 for rewarded pull
    actionLabels(ceil(missedPushTones./sampleDown)) = 5; % 5 for missedpushtone (MISS)
    actionLabels(ceil(missedPullTones./sampleDown)) = 6; % 6 for missedpulltone
    actionLabels(ceil(CorrectRejections./sampleDown)) = 7; % 7 for CORRECT REJECTION
    actionLabels(ceil(FalseAlarms./sampleDown)) = 8; % 8 for FALSE ALARM
    
    
    % figure for debugging
    % figure;
    % subplot(3,1,1)
    % plot(backLever)
    % hold on
    % plot(find(actionLabels == 1),zeros(size(find(actionLabels == 1))),'r*');
    % plot(find(actionLabels == 2),zeros(size(find(actionLabels == 2))),'g*');
    % subplot(3,1,2)
    % plot(frontLever)
    % hold on
    % plot(find(actionLabels == 1),zeros(size(find(actionLabels == 1))),'r*');
    % plot(find(actionLabels == 2),zeros(size(find(actionLabels == 2))),'g*');
    % subplot(3,1,3)
    % plot(Tonestim)
    % hold on
    % plot(find(actionLabels == 5),zeros(size(find(actionLabels == 5))),'k*');
    
    
    
    end
    
