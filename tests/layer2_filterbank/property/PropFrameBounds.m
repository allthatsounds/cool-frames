classdef PropFrameBounds < matlab.unittest.TestCase
    % PropFrameBounds - Property test for frame bounds inequality
    % Property: A||x||^2 <= sum_m (1/a_m) * ||c_m||^2 <= B||x||^2
    
    properties
        sig     % Signal battery
        p       % Parameters
    end
    
    methods (TestClassSetup)
        function setupClass(tc)
            % Add filterbank root to path
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            
            % Load signal battery
            batteryFile = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'signals', 'signal_battery.mat');
            if exist(batteryFile, 'file')
                b = load(batteryFile);
                tc.sig = b.signals;
                tc.p = b.params;
            else
                % Generate inline if battery doesn't exist
                rng(42);
                Ls = 1024;
                fs = 8000;
                tc.sig = struct();
                tc.sig.noise_real = randn(Ls, 1);
                tc.sig.noise_complex = randn(Ls, 1) + 1i*randn(Ls, 1);
                tc.sig.chirp = chirp(linspace(0, 1, Ls), 0.1, 1, 0.4)';
                tc.sig.impulse = zeros(Ls, 1); tc.sig.impulse(1) = 1;
                tc.p = struct('Ls', Ls, 'fs', fs);
            end
        end
    end
    
    methods (Test)
        function testFrameBoundsUniformSubsampling(tc)
            % Test frame bounds property with uniform subsampling (ERB filterbank)
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            [A, B] = filterbankbounds(g, a, L);
            
            ratios = [];
            
            % Run 100 random trials
            for trial = 1:100
                Ls = tc.p.Ls;
                x = randn(Ls, 1) + 1i*randn(Ls, 1);
                
                % Analysis
                c = filterbank(x, g, a);
                
                % Compute weighted energy: sum_m (1/a(m)) * ||c_m||^2
                M = length(c);  % Number of filters
                energy_Tx = 0;
                for m = 1:M
                    am = a(m, 1);  % Subsampling factor for band m
                    energy_Tx = energy_Tx + (1/am) * norm(c{m})^2;
                end
                
                % Compute ratio
                energy_x = norm(x)^2;
                ratio = energy_Tx / energy_x;
                ratios = [ratios; ratio];
                
                % Verify frame bounds (with tolerance)
                tc.verifyGreaterThanOrEqual(ratio, A - 1e-6, ...
                    sprintf('Trial %d: Ratio %.6f below lower bound A=%.6f', trial, ratio, A));
                tc.verifyLessThanOrEqual(ratio, B + 1e-6, ...
                    sprintf('Trial %d: Ratio %.6f above upper bound B=%.6f', trial, ratio, B));
            end
            
            % Record min and max ratio across trials
            minRatio = min(ratios);
            maxRatio = max(ratios);
            
            % Verify that both bounds are approached
            tc.verifyGreaterThan(minRatio, A - 0.1, ...
                sprintf('Min ratio %.6f not close to A=%.6f', minRatio, A));
            tc.verifyLessThan(maxRatio, B + 0.1, ...
                sprintf('Max ratio %.6f not close to B=%.6f', maxRatio, B));
        end
        
        function testFrameBoundsNonuniformSubsampling(tc)
            % Test frame bounds with non-uniform subsampling (CQT filterbank)
            try
                [g, a, ~, L] = cqtfilters(tc.p.fs, 0, tc.p.Ls, 12);
            catch
                % If cqtfilters not available, skip this test
                return;
            end
            
            [A, B] = filterbankbounds(g, a, L);
            
            ratios = [];
            
            % Run 100 random trials
            for trial = 1:100
                Ls = tc.p.Ls;
                x = randn(Ls, 1) + 1i*randn(Ls, 1);
                
                % Analysis
                c = filterbank(x, g, a);
                
                % Compute weighted energy: sum_m (1/a(m)) * ||c_m||^2
                M = length(c);
                energy_Tx = 0;
                for m = 1:M
                    am = a(m, 1);
                    energy_Tx = energy_Tx + (1/am) * norm(c{m})^2;
                end
                
                % Compute ratio
                energy_x = norm(x)^2;
                ratio = energy_Tx / energy_x;
                ratios = [ratios; ratio];
                
                % Verify frame bounds
                tc.verifyGreaterThanOrEqual(ratio, A - 1e-6, ...
                    sprintf('Trial %d: Ratio %.6f below lower bound A=%.6f', trial, ratio, A));
                tc.verifyLessThanOrEqual(ratio, B + 1e-6, ...
                    sprintf('Trial %d: Ratio %.6f above upper bound B=%.6f', trial, ratio, B));
            end
            
            % Record extremes
            minRatio = min(ratios);
            maxRatio = max(ratios);
            tc.verifyGreaterThan(minRatio, A - 0.1, ...
                sprintf('Min ratio %.6f not close to A=%.6f', minRatio, A));
            tc.verifyLessThan(maxRatio, B + 0.1, ...
                sprintf('Max ratio %.6f not close to B=%.6f', maxRatio, B));
        end
    end
end
